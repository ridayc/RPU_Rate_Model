#!/usr/bin/env python3
"""
analyze_snapshot.py

Load a saved Network snapshot and produce:
  - per-population firing rate histograms and maps
  - per-compartment weight/amp histograms and maps
  - per-compartment CV/C/CV maps & histograms (if available)
  - per-compartment band power fraction summaries (if available)

All plots go into subfolders:

  <outdir>/
    pop_<POP_ID>/
      ...
    comp_<CID>_<SRC>_to_<TGT>/
      ...

Usage:
  python analyze_snapshot.py \
      --snapshot snapshots/net_step_50000.pt \
      --outdir analysis_step_50000

Optional flags:
  --no-pop       : skip population plots
  --no-comp      : skip compartment plots
  --no-bands     : skip band-power-related plots
  --no-cv        : skip CV/C plots
"""

import os
import math
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from viz import (
    plot_volume,
    pop_volume,
    plot_population_rate_hist,
    plot_compartment_weight_hist,
    plot_compartment_amplitude_hist,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_network(snapshot_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    net = torch.load(snapshot_path, map_location=device)
    if hasattr(net, "device"):
        net.device = device
    return net


def safe_mean_tensor(t: torch.Tensor) -> float:
    t = t.detach().cpu()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return float(t.mean().item())


# ---------------------------------------------------------------------
# Summaries (printed)
# ---------------------------------------------------------------------

def summarize_populations(net):
    print("\n=== Population summaries ===")
    for pid, pop in net.populations.items():
        r = pop.rates.detach().cpu()
        mean_r = float(r.mean().item())
        std_r  = float(r.std(unbiased=False).item())
        cv_s   = std_r / (mean_r + 1e-8)

        # Approx CV_t using rate_average from first compartment if present
        r_avg = float("nan")
        cv_t  = float("nan")
        if len(pop.compartments) > 0:
            first_c = pop.compartments[next(iter(pop.compartments))]
            if hasattr(first_c, "rate_average"):
                ra = first_c.rate_average.detach().cpu()
                r_avg = float(ra.mean().item())
                r_std = float(ra.std(unbiased=False).item())
                cv_t  = r_std / (r_avg + 1e-8)

        print(
            f"Pop {pid:>3s} | mean rate = {mean_r:7.3f} | std = {std_r:7.3f} "
            f"| CV_s = {cv_s:7.3f} | rate_avg ≈ {r_avg:7.3f} | CV_t ≈ {cv_t:7.3f}"
        )
    print("================================\n")


def summarize_compartments(net):
    print("\n=== Compartment summaries ===")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            a = comp.a.detach().cpu()
            w = comp.w.detach().cpu()

            mean_a = float(a.mean().item())
            std_a  = float(a.std(unbiased=False).item())
            std_w  = float(w.std(unbiased=False).item())

            band_summary = ""
            if hasattr(comp, "rate_band") and isinstance(comp.rate_band, dict):
                # 'amplitude' or 'synapse' band power summary (out side)
                if "amplitude" in comp.rate_band:
                    amp = comp.rate_band["amplitude"]
                    P_f = amp["p"]["f"].detach().cpu()
                    P_m = amp["p"]["m"].detach().cpu()
                    P_s = amp["p"]["s"].detach().cpu()
                    P_tot = P_f + P_m + P_s + 1e-8
                    pf_med = float(torch.median(P_f / P_tot).item())
                    pm_med = float(torch.median(P_m / P_tot).item())
                    ps_med = float(torch.median(P_s / P_tot).item())
                    band_summary += f" | amp-bands: Pf_med={pf_med:5.2f}, Pm_med={pm_med:5.2f}, Ps_med={ps_med:5.2f}"

                if "synapse" in comp.rate_band and "out" in comp.rate_band["synapse"]:
                    syn_out = comp.rate_band["synapse"]["out"]
                    Pf = syn_out["p"]["f"].detach().cpu()
                    Pm = syn_out["p"]["m"].detach().cpu()
                    Ps = syn_out["p"]["s"].detach().cpu()
                    P_tot = Pf + Pm + Ps + 1e-8
                    pf_med = float(torch.median(Pf / P_tot).item())
                    pm_med = float(torch.median(Pm / P_tot).item())
                    ps_med = float(torch.median(Ps / P_tot).item())
                    band_summary += f" | syn-bands: Pf_med={pf_med:5.2f}, Pm_med={pm_med:5.2f}, Ps_med={ps_med:5.2f}"

            print(
                f"{cid:>10s} ({comp.sourceid}->{comp.targetid}) | "
                f"mean(a)={mean_a:7.3f} | std(a)={std_a:7.3f} | std(w)={std_w:7.3f}{band_summary}"
            )
    print("================================\n")


# ---------------------------------------------------------------------
# Population plots
# ---------------------------------------------------------------------

def plot_population_outputs(net, outdir: str):
    """
    For each population:
      - rate histogram
      - rate map (mean over depth)
      - grid of slices (if Z>1)
    All saved in subfolder: outdir/pop_<POP_ID>/
    """
    for pid, pop in net.populations.items():
        pop_dir = ensure_dir(os.path.join(outdir, f"pop_{pid}"))

        # 1) Histogram
        fig, ax = plot_population_rate_hist(
            net, pid,
            bins=60,
            range=None,
            figsize=(4, 3),
            title=f"{pid} rate histogram",
        )
        fig.savefig(os.path.join(pop_dir, "rate_hist.png"), dpi=150)
        plt.close(fig)

        # 2) Map (mean over depth)
        vol = pop_volume(net, pid, field="rates")  # (W,H,Z)
        fig, ax = plot_volume(
            vol,
            mode="func",
            func=lambda v: np.mean(v, axis=2),
            normalize=False,
            title=f"{pid} mean rate over depth",
            figsize=(4, 4),
            cmap="viridis",
        )
        fig.savefig(os.path.join(pop_dir, "rate_map_mean.png"), dpi=150)
        plt.close(fig)

        # 3) All slices grid
        W, H, Z = pop.size
        if Z > 1:
            fig, axes = plot_volume(
                vol,
                mode="grid",
                normalize=False,
                title=f"{pid} rate slices (all z)",
                figsize=None,
                cmap="viridis",
            )
            fig.savefig(os.path.join(pop_dir, "rate_slices_grid.png"), dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------
# Compartment plots: weights, amplitudes, CV/C, bands
# ---------------------------------------------------------------------

def plot_compartment_basic(comp, comp_dir: str):
    """
    Plots:
      - weight histogram
      - amplitude histogram
      - amplitude map (mean over z in target geometry)
    """
    # 1) weight histogram
    fig, ax = plot_compartment_weight_hist(
        comp,
        bins=60,
        logx=False,
        logy=False,
        title=f"Weights {comp.sourceid}->{comp.targetid} ({comp.id})",
    )
    fig.savefig(os.path.join(comp_dir, "weights_hist.png"), dpi=150)
    plt.close(fig)

    # 2) amplitude histogram
    fig, ax = plot_compartment_amplitude_hist(
        comp,
        bins=60,
        logx=False,
        logy=False,
        title=f"Amplitudes {comp.sourceid}->{comp.targetid} ({comp.id})",
    )
    fig.savefig(os.path.join(comp_dir, "amps_hist.png"), dpi=150)
    plt.close(fig)

    # 3) amplitude map (mean over z)
    a = comp.a.detach().cpu().numpy()
    W_t, H_t, Z_t = comp.target.size
    vol = a.reshape(W_t, H_t, Z_t)

    fig, ax = plot_volume(
        vol,
        mode="func",
        func=lambda v: np.mean(v, axis=2),
        normalize=False,
        title=f"amp a (mean over z) {comp.sourceid}->{comp.targetid}",
        figsize=(4, 4),
        cmap="viridis",
    )
    fig.savefig(os.path.join(comp_dir, "amp_map_mean.png"), dpi=150)
    plt.close(fig)


def plot_compartment_cv_and_corr(comp, comp_dir: str):
    """
    If comp.stat is True, plot:
      - histograms: CVt, CVs, C, |C|
      - maps (mean over z): CVt, CVs, C
    """
    if not getattr(comp, "stat", False):
        return

    # All are per-target (length = nneu_target)
    CVt = comp.CVt.detach().cpu()
    CVs = comp.CVs.detach().cpu()
    C   = comp.C.detach().cpu()
    C2  = comp.C2.detach().cpu()  # abs C smoothed

    # Histograms
    def _hist(vals, fname, title, xlabel):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        v = vals.numpy()
        ax.hist(v, bins=60)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, fname), dpi=150)
        plt.close(fig)

    _hist(CVt, "CVt_hist.png", f"CVt ({comp.id})", "CVt")
    _hist(CVs, "CVs_hist.png", f"CVs ({comp.id})", "CVs")
    _hist(C,   "C_hist.png",   f"C ({comp.id})",   "C")
    _hist(torch.abs(C), "C_abs_hist.png", f"|C| ({comp.id})", "|C|")

    # Maps (mean over z) in target geometry
    W_t, H_t, Z_t = comp.target.size
    # reshape to (W,H,Z) even though Z dimension is trivial; we treat as 1-depth volume
    vol_CVt = CVt.numpy().reshape(W_t, H_t, Z_t)
    vol_CVs = CVs.numpy().reshape(W_t, H_t, Z_t)
    vol_C   = C.numpy().reshape(W_t, H_t, Z_t)

    for vol, name, title in [
        (vol_CVt, "CVt_map_mean.png", f"CVt map (mean over z) {comp.id}"),
        (vol_CVs, "CVs_map_mean.png", f"CVs map (mean over z) {comp.id}"),
        (vol_C,   "C_map_mean.png",   f"C map (mean over z) {comp.id}"),
    ]:
        fig, ax = plot_volume(
            vol,
            mode="func",
            func=lambda v: np.mean(v, axis=2),
            normalize=False,
            title=title,
            figsize=(4, 4),
            cmap="viridis",
        )
        fig.savefig(os.path.join(comp_dir, name), dpi=150)
        plt.close(fig)


def plot_compartment_band_power(comp, comp_dir: str):
    """
    If comp.rate_band has 'synapse' or 'amplitude',
    plot band power fraction histograms & maps (for 'out' or 'amplitude').
    """
    if not hasattr(comp, "rate_band") or not isinstance(comp.rate_band, dict):
        return

    # Helper for plotting band fractions for a given band dict
    def _plot_band_dict(prefix, band_dict, size, fname_suffix):
        # band_dict has band_dict["p"]["f"/"m"/"s"] etc.
        Pf = band_dict["p"]["f"].detach().cpu()
        Pm = band_dict["p"]["m"].detach().cpu()
        Ps = band_dict["p"]["s"].detach().cpu()
        Ptot = Pf + Pm + Ps + 1e-8

        frac_f = Pf / Ptot
        frac_m = Pm / Ptot
        frac_s = Ps / Ptot

        # Hist
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(frac_f.numpy(), bins=50, alpha=0.5, label="fast")
        ax.hist(frac_m.numpy(), bins=50, alpha=0.5, label="mid")
        ax.hist(frac_s.numpy(), bins=50, alpha=0.5, label="slow")
        ax.set_xlabel("power fraction")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title(f"{prefix} band power fractions ({comp.id})")
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, f"{fname_suffix}_power_frac_hist.png"), dpi=150)
        plt.close(fig)

        # Maps (mean over z)
        W, H, Z = size
        for frac, tag in [(frac_f, "fast"), (frac_m, "mid"), (frac_s, "slow")]:
            vol = frac.numpy().reshape(W, H, Z)
            fig, ax = plot_volume(
                vol,
                mode="func",
                func=lambda v: np.mean(v, axis=2),
                normalize=False,
                title=f"{prefix} {tag} power fraction map (mean over z) {comp.id}",
                figsize=(4, 4),
                cmap="viridis",
            )
            fig.savefig(os.path.join(comp_dir, f"{fname_suffix}_power_frac_map_{tag}.png"), dpi=150)
            plt.close(fig)

    # amplitude-based bands (per target)
    if "amplitude" in comp.rate_band:
        amp = comp.rate_band["amplitude"]
        size = comp.target.size  # per-target bands
        _plot_band_dict("amplitude", amp, size, "amp_bands")

    # synapse-based bands: in/out
    if "synapse" in comp.rate_band:
        syn = comp.rate_band["synapse"]
        if "out" in syn:
            size = comp.target.size
            _plot_band_dict("syn-out", syn["out"], size, "syn_out_bands")
        if "in" in syn:
            size = comp.source.size
            _plot_band_dict("syn-in", syn["in"], size, "syn_in_bands")


def plot_compartments(net, outdir: str, do_cv: bool = True, do_bands: bool = True):
    """
    For each compartment, create subfolder:
      comp_<cid>_<src>_to_<tgt>/
    and fill with:
      - basic weight/amp plots
      - CV/C/CV maps (if do_cv and comp.stat)
      - band power plots (if do_bands and bands exist)
    """
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            comp_dirname = f"comp_{cid}_{comp.sourceid}_to_{comp.targetid}"
            comp_dir = ensure_dir(os.path.join(outdir, comp_dirname))

            # Always basic
            plot_compartment_basic(comp, comp_dir)

            # Optional CV/C/CV
            if do_cv:
                plot_compartment_cv_and_corr(comp, comp_dir)

            # Optional bands
            if do_bands:
                plot_compartment_band_power(comp, comp_dir)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a saved Network snapshot: per-population and per-compartment plots."
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to .pt snapshot saved by net.save() or torch.save(net,...).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading network ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for plots. Default: <snapshot_basename>_analysis",
    )
    parser.add_argument(
        "--no-pop",
        action="store_true",
        help="Skip population plots.",
    )
    parser.add_argument(
        "--no-comp",
        action="store_true",
        help="Skip compartment plots.",
    )
    parser.add_argument(
        "--no-bands",
        action="store_true",
        help="Skip band power plots for compartments.",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip CV/C plots for compartments.",
    )

    args = parser.parse_args()

    if args.outdir is None:
        base = os.path.splitext(os.path.basename(args.snapshot))[0]
        outdir = base + "_analysis"
    else:
        outdir = args.outdir
    ensure_dir(outdir)

    print(f"Loading network from: {args.snapshot}")
    net = load_network(args.snapshot, device=args.device)

    # Textual summaries
    summarize_populations(net)
    summarize_compartments(net)

    # Plots
    if not args.no_pop:
        print("Generating population plots...")
        plot_population_outputs(net, outdir)

    if not args.no_comp:
        print("Generating compartment plots...")
        plot_compartments(net, outdir, do_cv=not args.no_cv, do_bands=not args.no_bands)

    print(f"All plots written under: {outdir}")
    print("Done.")


if __name__ == "__main__":
    main()