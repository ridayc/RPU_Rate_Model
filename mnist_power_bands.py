# mnist_train.py
import math
import copy as cp
import os
import torch
import torchvision as tv
import torchvision.transforms as T

from network import (
    Network,
    population_parameters,
    compartment_parameters,
)

# -------------------------
# Helpers
# -------------------------

def make_sizes_ei(size_e, frac_i=0.28):
    """Given excitatory size [W,H,Z], build inhibitory size with same W,H and ~frac_i*Z layers."""
    W, H, Z = size_e
    Zi = max(1, int(round(frac_i * Z)))
    return [W, H, Zi]

def ema_update(x, new, alpha):
    return (1 - alpha) * x + alpha * new

# Normalize Hebbian LR by expected rate scale (pre_r0 * post_r0)
def eta_scaled(eta_base, r0_pre, r0_post, eps=1e-6):
    return eta_base / max(r0_pre * r0_post, eps)


# -------------------------
# Build network (P, E, I only)
# -------------------------
def build_net(
    device="cuda" if torch.cuda.is_available() else "cpu",
    Z_E=8,
    frac_i=0.28
):
    device = torch.device(device)

    # Shapes
    size_P = [28, 28, 1]
    size_E = [28, 28, Z_E]
    size_I = make_sizes_ei(size_E, frac_i=frac_i)

    # --- Population definitions ---
    R_E = 70.0
    R_I = 150.0

    pops = {}
    pops["P"] = population_parameters(
        "P",
        size=size_P,
        tau=1,
        rate_inflection=255.0,
        activation_exponent=1.0,
        baseline=0.0,
        cap=600.0,
        activation=None,
    )
    pops["E"] = population_parameters(
        "E",
        size=size_E,
        tau=3,
        rate_inflection=R_E * 1.,
        activation_exponent=0.85,
        baseline=1.0,
        cap=600.0,
        activation=None,
    )
    pops["I"] = population_parameters(
        "I",
        size=size_I,
        tau=2,
        rate_inflection=R_I * 1.5,
        activation_exponent=1.3,
        baseline=0.0,
        cap=1000.0,
        activation=None,
    )

    # --- Global timescales / learning rates ---
    AVG_TAU  = 900             # long-term averages for amplitude, etc.
    TAUS = AVG_TAU*4
    AVG_TAU2 = 60              # covariance / Hebb averages
    DELTA_E  = 0.0001 / AVG_TAU   # excitatory amp learning
    DELTA_I = 0.00005 / TAUS        # inhibitory amp learning (off for now)
    RHO      = 0.00 / TAUS 

    # Base Hebbian LR scale
    BASE  = 0.05 * 0.001  # same as in your template
    LR_EE = 2 * BASE
    LR_EI = 1 * BASE
    LR_IE = 8 * BASE
    LR_II = 1 * BASE
    BETA_E = BASE * 0.0001
    BETA_I = BASE * 0.001

    # Pull r0s for eta scaling
    r0P = pops["P"]["r0"]
    r0E = pops["E"]["r0"]
    r0I = pops["I"]["r0"]

    # frequency power bands for synaptic learning (E-E)
    E_E_band = {}
    E_E_band["synapse"] = {}
    E_E_band["synapse"]["in"] = {}
    E_E_band["synapse"]["out"] = {}
    freq = {"f":7,"m":20,"s":60}
    taup = TAUS
    theta = {"f":[0.2,0.35],"s":[0.2,0.6]}
    eta_out = {"f":[1.,2.],"s":[2.,1.]}
    eta_in = {"f":[0.3,0.1],"s":[0.1,0.3]}
    for i in ["in","out"]:
        E_E_band["synapse"][i]["tau"] = cp.copy(freq)
        E_E_band["synapse"][i]["taup"] = taup
        E_E_band["synapse"][i]["theta"] = cp.deepcopy(theta)
    E_E_band["synapse"]["in"]["eta"] = cp.deepcopy(eta_in)
    E_E_band["synapse"]["out"]["eta"] = cp.deepcopy(eta_out)

    # frequency power bands for amplitude learning (I-E)
    I_E_band = {}
    I_E_band["amplitude"] = {}
    freq = {"f":7,"m":20,"s":60}
    taup = TAUS
    eta = {"f":[4./TAUS*0.0,2./TAUS*0.0],"s":[1./TAUS*0.0,2./TAUS*0.0]}
    I_E_band["amplitude"]["target"] = "I_E"
    I_E_band["amplitude"]["tau"] = cp.copy(freq)
    I_E_band["amplitude"]["taup"] = taup
    I_E_band["amplitude"]["theta"] = cp.deepcopy(theta)
    I_E_band["amplitude"]["eta"] = cp.deepcopy(eta)

    frequencies = {}
    frequencies["theta"] = {"period": 11, "tau": 16, "alpha": LR_EE/r0E/r0E/3*0}
    #frequencies["gamma"] = {"period": 4, "tau": 6, "alpha": LR_EE/R_E/R_E/3*1}
    #frequencies["quick"] = {"period": 3, "tau": 4, "alpha": LR_EE/R_E/R_E/3*1}

    # --- Compartments dict ---
    comps = {}

    # 1) P -> E: feedforward, fixed-ish
    comps["P_E"] = compartment_parameters(
        id="P_E",
        source="P",
        target="E",
        ellipse=[3, 3],
        tsyn=16,
        A=0.4,
        A0=0.4,
        eta=eta_scaled(LR_EE, r0P, r0E) * 0.1,  # small plasticity
        nu=0.0,
        beta=0.0,
        bands=None,
        rho=DELTA_E*0.0,
        tau=AVG_TAU,
        taug = TAUS,
        thetaz = 2.,
        z_value = 1./3,
        zeta = DELTA_E*0.1,
        ratio = "ueff",
        rin=0.0,
        rout=0.0,
        tauin=-AVG_TAU2,
        tauout=-AVG_TAU2,
        delta=0.0,              # no amplitude learning on feedforward for now
        rate_target=R_E,
        eps=5.0,
        stype="",               # not used for E/PV statistics
    )

    # 2) E -> E: recurrent excitatory, tracked by PV (stype="E")
    comps["E_E"] = compartment_parameters(
        id="E_E",
        source="E",
        target="E",
        ellipse=[4, 4],
        tsyn=18,
        A=4,
        A0=4,
        eta=eta_scaled(LR_EE, r0E, r0E)*1,
        alpha=eta_scaled(LR_EE, r0P, r0E) * 0.0,
        nu=0.0,
        beta=BETA_E,
        bands=E_E_band,
        freq=frequencies,
        rho=RHO,
        tau=AVG_TAU,
        rin=.0,
        rout=.0,
        tauin=-AVG_TAU2,
        tauout=-AVG_TAU2,
        delta=DELTA_E,
        rate_target=R_E,
        eps=5.0,
        stype="",
        stat = True,
        power={
            "tauf": AVG_TAU,   # fast mixing timescale
            "taus": TAUS,      # slow mixing timescale
        }
    )

    # 3) E -> I: excitatory drive to inhibitory
    comps["E_I"] = compartment_parameters(
        id="E_I",
        source="E",
        target="I",
        ellipse=[4, 4],
        tsyn=18,
        A=5.5,
        A0=5.5,
        eta=eta_scaled(LR_EI, r0E, r0I)*1,
        nu=0.0,
        beta=BETA_E,
        bands=None,
        rho=RHO,
        tau=AVG_TAU,
        rin=0.0,
        rout=0.0,
        tauin=-AVG_TAU2,
        tauout=-AVG_TAU2,
        delta=DELTA_E * 0.1,   # small amplitude plasticity
        rate_target=R_I,
        eps=5.0,
        stype="",              # not tracked by PV statistics
    )

    # 4) I -> E: inhibitory PV-like compartment (stype="PV")
    comps["I_E"] = compartment_parameters(
        id="I_E",
        source="I",
        target="E",
        ellipse=[3, 3],
        tsyn=18,
        A=-8.5,               # inhibitory
        A0=8.5,               # target amplitude (magnitude)
        eta=eta_scaled(LR_IE, r0E, r0I),
        nu=0.0,
        beta=BETA_I,
        bands=I_E_band,
        rho=1/TAUS*0.0000,              # you might later add rho for loga regularization
        tau=AVG_TAU,
        taug=TAUS,
        rin=0.0,
        tauin=-AVG_TAU2,
        rout=1.5*R_E,
        zeta=DELTA_I,
        z_value = -0.25,
        ratio = "E/I",
        tauout=-AVG_TAU2,
        delta=0,        # inhibitory amplitude learning (off now)
        rate_target=R_E,
        eps=5.0,
        stype="",
    )

    # 5) I -> I: inhibitory recurrent, not PV-tracked
    comps["I_I"] = compartment_parameters(
        id="I_I",
        source="I",
        target="I",
        ellipse=[3, 3],
        tsyn=18,
        A=-3.0,
        A0=3.0,
        eta=eta_scaled(LR_II, r0I, r0I),
        nu=0.0,
        beta=BETA_I*0.01,
        bands=None,
        rho=RHO,
        tau=AVG_TAU,
        rin=0.0,
        tauin=-AVG_TAU2,
        rout=3 * R_I,
        tauout=-AVG_TAU2,
        zeta=DELTA_I,
        z_value = -0.5,
        ratio = "E/I",
        delta=0,
        rate_target=R_I,
        eps=5.0,
        stype="",
    )

    net = Network(device, pops, comps)
    return net


# -------------------------
# Logging helpers
# -------------------------
def log_population_stats(net):
    print("\n--- Population Activity Summary ---")
    for pid, pop in net.populations.items():
        r = pop.rates.detach().cpu()
        mean_r = float(r.mean().item())
        std_r  = float(r.std(unbiased=False).item())
        CV_s   = std_r / (mean_r + 1e-8)

        # approximate temporal CV using first compartment's rate_average
        r_avg = 0.0
        CV_t  = 0.0
        if len(pop.compartments) > 0:
            first_c = pop.compartments[next(iter(pop.compartments))]
            ra = first_c.rate_average.detach().cpu()
            r_avg = float(ra.mean().item())
            r_std = float(ra.std(unbiased=False).item())
            CV_t  = r_std / (r_avg + 1e-8)

        print(f"Pop {pid:>3s} | mean r = {mean_r:7.3f} | CV_s = {CV_s:7.3f} | r_avg = {r_avg:7.3f} | CV_t ≈ {CV_t:7.3f}")
    print("----------------------------------\n")


def log_compartment_stats(net):
    
    print("\n--- Compartment CV / C stats (E & PV) ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            if not comp.stat:
                continue

            CVt = comp.CVt.detach().cpu()
            CVs = comp.CVs.detach().cpu()
            C   = comp.C.detach().cpu()
            C2  = comp.C2.detach().cpu()
            r = comp.lrates.detach().cpu()

            def safe_mean(x):
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                return float(x.mean().item())

            print(
                f"[{pid}:{cid:>4s}] "
                f"CVt_med={torch.median(CVt):6.3f}  "
                f"CVs_med={torch.median(CVs):6.3f}  "
                f"C_med={torch.median(C):7.4f}  "
                f"C2_mean={safe_mean(C2):7.4f}"
            )

    print("----------------------------------------\n")

    print("\n--- Compartment Power ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            if "synapse" not in comp.rate_band:
                continue

            Pf = comp.rate_band["synapse"]["out"]["p"]["f"].detach().cpu()
            Pm = comp.rate_band["synapse"]["out"]["p"]["m"].detach().cpu()
            Ps = comp.rate_band["synapse"]["out"]["p"]["s"].detach().cpu()
            Ptot = Pf+Pm+Ps+1e-8

            def safe_mean(x):
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                return float(x.mean().item())

            print(
                f"[{pid}:{cid:>4s}] "
                f"Pf={torch.median(Pf/Ptot):6.3f}  "
                f"Pm={torch.median(Pm/Ptot):6.3f}  "
                f"Ps={torch.median(Ps/Ptot):7.4f}  "
            )

    print("----------------------------------------\n")

    print("\n--- Network Weight Summary ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            a = comp.a.detach().cpu()
            w = comp.w.detach().cpu()
            # assuming numerator/denominator exist in your Compartment implementation
            ES = comp.numerator.detach().cpu()
            IS = comp.denominator.detach().cpu()
            G = torch.mean(ES/(IS+1e-8))
            
            mean_a = a.mean().item()
            std_a = a.std(unbiased=False).item()
            std_w = w.std(unbiased=False).item()

            print(f"{(comp.sourceid+'-'+comp.targetid):>8s} | mean amp = {mean_a:7.3f}  | std amp = {std_a:7.4f}  |  std w = {(std_w):7.5f}  |  I-E = {(G):7.5f}")
    print("--------------------------------\n")
    

def P_E_corr(net):
    """Crude instantaneous spatial correlation between input P and mean E activity over depth."""
    P_pop = net.populations["P"]
    E_pop = net.populations["E"]

    # P: (28*28*1,)
    P = P_pop.rates.detach()

    # E: (W*H*Z,)
    W, H, Z = E_pop.size
    E_rates = E_pop.rates.detach().view(W, H, Z)

    # Average E over depth Z -> (W,H)
    E_map = E_rates.mean(dim=2).reshape(-1)

    # Make sure devices match
    E_map = E_map.to(P.device)

    # Zero-mean
    P_c = P - P.mean()
    E_c = E_map - E_map.mean()

    num = (P_c * E_c).mean()
    denom = P_c.std(unbiased=False) * E_c.std(unbiased=False) + 1e-8

    return float(num / denom)


# -------------------------
# MNIST "training" (unsupervised)
# -------------------------
def train_mnist_unsupervised(
    epochs=1,
    steps_per_img=10,
    warm_up_per_img=10,
    device=None,
    batch_size=1,
    seed=123,
    log_every=500,
    snapshot_dir=None,
    snapshot_every=None,
    snapshot_prefix="mnist_net",
):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # if we want snapshots, ensure directory exists
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        if snapshot_every is None:
            snapshot_every = log_every

    transform = T.Compose([T.ToTensor()])
    trainset = tv.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    net = build_net(device=device)

    img_idx_global = 0
    for epoch in range(epochs):
        for (imgs, labels) in loader:
            img   = imgs[0, 0] * 255.0
            label = int(labels[0].item())  # not used yet

            # Set input rates
            P = net.populations["P"]
            P.rates[:] = img.flatten().clone().to(net.device)

            # Warmup with this input
            for _ in range(warm_up_per_img):
                net.iterate()
            # Active steps with this input
            for _ in range(steps_per_img):
                P.rates[:] = img.flatten().clone().to(net.device)
                net.iterate()

            # Logging
            if img_idx_global % log_every == 0:
                print(f"\n=== Epoch {epoch}, image {img_idx_global}, label={label} ===")
                log_population_stats(net)
                log_compartment_stats(net)
                print(f"Instant P–E corr: {P_E_corr(net):+.3f}")
                print("=========================================\n")

            # Snapshots
            if snapshot_dir is not None and snapshot_every is not None:
                if img_idx_global % snapshot_every == 0:
                    fname = f"{snapshot_prefix}_e{epoch}_i{img_idx_global}.pt"
                    path = os.path.join(snapshot_dir, fname)
                    net.save(path)
                    print(f"[snapshot] Saved network to {path}")

            img_idx_global += 1

    # Final snapshot
    if snapshot_dir is not None:
        fname = f"{snapshot_prefix}_final.pt"
        path = os.path.join(snapshot_dir, fname)
        net.save(path)
        print(f"[snapshot] Saved final network to {path}")

    return net


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unsupervised MNIST drive for recurrent E/I network."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-img", type=int, default=5)
    parser.add_argument("--warm_up_per_img", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--log-every", type=int, default=500)

    # new CLI snapshot options
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help="Directory to save network snapshots (if not set, no snapshots are saved).")
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help="Save a snapshot every N images. Defaults to log_every if not set.")
    parser.add_argument("--snapshot-prefix", type=str, default="mnist_net",
                        help="Filename prefix for snapshot files.")

    args = parser.parse_args()

    _ = train_mnist_unsupervised(
        epochs=args.epochs,
        steps_per_img=args.steps_per_img,
        warm_up_per_img=args.warm_up_per_img,   # fixed bug: use the actual arg
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        log_every=args.log_every,
        snapshot_dir=args.snapshot_dir,
        snapshot_every=args.snapshot_every,
        snapshot_prefix=args.snapshot_prefix,
    )