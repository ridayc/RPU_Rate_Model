# ============================================================
# Logging helpers
# ============================================================

import math
import torch

from network import row_sum

def _safe_mean(x):
    """Mean with nan/inf protection."""
    return float(torch.nan_to_num(x, nan=0., posinf=0., neginf=0.).mean().item())


def log_population_stats(net):
    """Print mean firing rate and spatial/temporal CV for all populations."""
    print("\n--- Population Activity Summary ---")
    for pid, pop in net.populations.items():
        r     = pop.rates.detach().cpu()
        mean_r = float(r.mean())
        std_r  = float(r.std(unbiased=False))
        CV_s   = std_r / (mean_r + 1e-8)

        # Temporal CV estimated from first compartment's rate_average
        r_avg = CV_t = 0.
        if pop.compartments:
            first_c = next(iter(pop.compartments.values()))
            ra    = first_c.rate_average.detach().cpu()
            r_avg = float(ra.mean())
            CV_t  = float(ra.std(unbiased=False)) / (r_avg + 1e-8)

        print(
            f"Pop {pid:>3s} | "
            f"mean r = {mean_r:7.3f} | "
            f"CV_s = {CV_s:7.3f} | "
            f"r_avg = {r_avg:7.3f} | "
            f"CV_t ≈ {CV_t:7.3f}"
        )
    print("----------------------------------\n")


def log_compartment_stats(net):
    """
    Print compartment-level diagnostics:
      - CV and correlation stats for tracked compartments (stat=True)
      - Band power fractions for compartments with band tracking
      - SST quantile thresholds
      - Weight distribution summary for all compartments
    """
    # --- Correlation / CV stats ---
    tracked = [(pid, cid, comp)
               for pid, pop in net.populations.items()
               for cid, comp in pop.compartments.items()
               if comp.stat]
    if tracked:
        print("\n--- Compartment CV / C stats ---")
        for pid, cid, comp in tracked:
            CVt = comp.CVt.detach().cpu()
            CVs = comp.CVs.detach().cpu()
            C   = comp.C.detach().cpu()
            C2  = comp.C2.detach().cpu()
            m   = comp.rit_slow.detach().cpu()
            print(
                f"[{pid}:{cid:>4s}] "
                f"CVt_med={torch.median(CVt / (m + 1e-8)):6.3f}  "
                f"CVs_med={torch.median(CVs):6.3f}  "
                f"C_med={torch.median(C):7.4f}  "
                f"C2_mean={_safe_mean(C2):7.4f}"
            )
        print("--------------------------------\n")

    # --- Band power and SST stats ---
    band_or_sst = [(pid, cid, comp)
                   for pid, pop in net.populations.items()
                   for cid, comp in pop.compartments.items()
                   if comp.rate_band or comp.SST is not None]
    if band_or_sst:
        print("\n--- Band Power / SST quantiles ---")
        for pid, cid, comp in band_or_sst:
            for band_key in ("synapse", "amplitude"):
                if band_key not in comp.rate_band:
                    continue
                band_out_key = "out" if band_key == "synapse" else None
                rb = comp.rate_band[band_key]
                p_dict = rb["out"]["p"] if band_out_key else rb["p"]
                Pf = p_dict["f"].detach().cpu()
                Pm = p_dict["m"].detach().cpu()
                Ps = p_dict["s"].detach().cpu()
                Ptot = Pf + Pm + Ps + 1e-8
                print(
                    f"[{pid}:{cid:>4s}][{band_key}] "
                    f"Pf={torch.median(Pf/Ptot):6.3f}  "
                    f"Pm={torch.median(Pm/Ptot):6.3f}  "
                    f"Ps={torch.median(Ps/Ptot):6.3f}"
                )
        print("----------------------------------\n")

    # --- Weight distribution summary ---
    print("\n--- Network Weight Summary ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            a    = comp.a.detach().cpu()
            w    = comp.w.detach().cpu()
            wind = comp.w_ind[0, :].detach().cpu()
            wq   = comp.wq.detach().cpu()

            if comp.ratio in ("corr", "NMC"):
                G = float(comp.numerator.median())
            else:
                G = float((comp.numerator / (comp.denominator + 1e-8)).median())

            rat   = math.exp(-float(comp.dM.detach().cpu().median()))
            Neff  = float((1. / comp.k * row_sum((w < wq[wind]).float(), wind)).mean())
            bfact = math.exp(float(comp.dN.detach().cpu().median()))
            mean_a = float(a.mean())
            std_a  = float(a.std(unbiased=False))
            m_w    = float(w.mean())
            std_w  = float(w.std(unbiased=False))

            print(
                f"{(comp.sourceid+'-'+comp.targetid):>8s} | "
                f"A_m = {mean_a:7.3f} | "
                f"A_s = {std_a:12.3e} | "
                f"CV(w) = {std_w / (m_w + 1e-8):7.3f} | "
                f"an/ap = {rat:7.5f} | "
                f"N = {Neff:7.5f} | "
                f"b = {bfact:12.3e} | "
                f"I-E = {G:7.5f}"
            )
    print("--------------------------------\n")


def log_P_E_correlation(net):
    """Instantaneous spatial correlation between pixel input P and mean E activity."""
    P_rates = net.populations["P"].rates.detach()
    W, H, Z = net.populations["E"].size
    E_map   = net.populations["E"].rates.detach().view(W, H, Z).mean(dim=2).reshape(-1)
    E_map   = E_map.to(P_rates.device)
    P_c = P_rates - P_rates.mean()
    E_c = E_map   - E_map.mean()
    corr = (P_c * E_c).mean() / (P_c.std(unbiased=False) * E_c.std(unbiased=False) + 1e-8)
    print(f"Instant P–E corr: {float(corr):+.3f}")