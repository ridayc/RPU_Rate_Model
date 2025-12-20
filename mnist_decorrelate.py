# mnist_train.py
import math
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
        rate_inflection=R_E * 1.5,
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
        activation_exponent=1.5,
        baseline=0.0,
        cap=1000.0,
        activation=None,
    )

    # --- Global timescales / learning rates ---
    AVG_TAU  = 900             # long-term averages for amplitude, etc.
    TAUS = AVG_TAU*4
    AVG_TAU2 = 30              # covariance / Hebb averages
    DELTA_E  = 0.1 / AVG_TAU   # excitatory amp learning
    DELTA_I  = 0.0           # inhibitory amp learning (off for now)
    RHO      = 0.00 / TAUS 

    # Base Hebbian LR scale
    BASE  = 0.05 * 0.1  # same as in your template
    LR_EE = 2 * BASE
    LR_EI = 1 * BASE
    LR_IE = 8 * BASE
    LR_II = 1 * BASE
    BETA_E = BASE * 0.001
    BETA_I = BASE * 0.1

    # Pull r0s for eta scaling
    r0P = pops["P"]["r0"]
    r0E = pops["E"]["r0"]
    r0I = pops["I"]["r0"]

    # Frequency bands: keep OFF for now (no oscillation-specific learning)
    bands_none = {}
    bands_e1 = {}
    bands_e1["theta"] = {"period": 11, "tau": 16, "alpha": LR_EE/R_E/R_E/3*1}
    bands_e1["gamma"] = {"period": 4, "tau": 6, "alpha": LR_EE/R_E/R_E/3*1}
    bands_e1["quick"] = {"period": 3, "tau": 4, "alpha": LR_EE/R_E/R_E/3*1}

    # --- Compartments dict ---
    comps = {}

    # 1) P -> E: feedforward, fixed-ish
    comps["P_E"] = compartment_parameters(
        id="P_E",
        source="P",
        target="E",
        ellipse=[3, 3],
        tsyn=16,
        A=2.4,
        A0=2.4,
        eta=eta_scaled(LR_EE, r0P, r0E) * 0.01,  # small plasticity
        nu=0.0,
        beta=0.0,
        bands=bands_none,
        rho=DELTA_E*0.01,
        tau=AVG_TAU,
        rin=0.0,
        rout=0.0,
        tauin=-AVG_TAU2,
        tauout=-AVG_TAU2,
        delta=0.0,              # no amplitude learning on feedforward for now
        rate_target=R_E,
        eps=5.0,
        stype="",               # not used for E/PV statistics
        pooling={}
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
        nu=0.0,
        beta=BETA_E,
        bands=bands_none,
        rho=RHO,
        tau=AVG_TAU,
        rin=0.0,
        rout=0.0,
        tauin=-AVG_TAU2,
        tauout=-AVG_TAU2,
        delta=DELTA_E,
        rate_target=R_E,
        eps=5.0,
        stype="E",
        pooling={
            "tauf": AVG_TAU,   # fast mixing timescale
            "taus": TAUS,  # slow mixing timescale
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
        eta=eta_scaled(LR_EI, r0E, r0I),
        nu=0.0,
        beta=BETA_E,
        bands=bands_none,
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
        pooling={}
    )

    # 4) I -> E: inhibitory PV-like compartment (stype="PV")
    comps["I_E"] = compartment_parameters(
        id="I_E",
        source="I",
        target="E",
        ellipse=[3, 3],
        tsyn=18,
        A=-5.5,               # inhibitory
        A0=5.5,               # target amplitude (magnitude)
        eta=eta_scaled(LR_IE, r0E, r0I),
        nu=0.0,
        beta=BETA_I,
        bands=bands_none,
        rho=1/TAUS*0.0000,              # you might later add rho for loga regularization
        tau=AVG_TAU,
        rin=0.0,
        tauin=-AVG_TAU2,
        rout=3 * R_E,
        tauout=-AVG_TAU2,
        delta=DELTA_I,        # inhibitory amplitude learning (off now)
        rate_target=R_E,
        eps=5.0,
        stype="PV",
        pooling={
            "etaA": 1./TAUS*0.001,     # upward direction step size inside Region A
            "etaB": 5./TAUS*0.001,     # outside Region A (you can differentiate later)
            "etaC": 0.0/TAUS*0.1,
            "skip": AVG_TAU,      # minimal waiting scale for direction checks
            "theta_t": 0.3,   # CVt threshold
            "theta_s": 0.8,   # CVs threshold
        }
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
        beta=BETA_I,
        bands=bands_none,
        rho=RHO,
        tau=AVG_TAU,
        rin=0.0,
        tauin=-AVG_TAU2,
        rout=3 * R_I,
        tauout=-AVG_TAU2,
        delta=DELTA_I,
        rate_target=R_I,
        eps=5.0,
        stype="",
        pooling={}
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
            if comp.stype not in ["E", "PV"]:
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

    print("\n--- Network Weight Summary ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            a = comp.a.detach().cpu()
            w = comp.w.detach().cpu()
            

            mean_a = a.mean().item()
            std_a = a.std(unbiased=False).item()
            std_w = w.std(unbiased=False).item()

            print(f"{comp.sourceid+"-"+comp.targetid:>8s} | mean amp = {mean_a:7.3f}  | std amp = {std_a:7.4f}  |  std w = {(std_w):7.5f}")
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
):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

            # Run network for a few steps with this input
            for _ in range(warm_up_per_img):
                net.iterate()
            # Run network for a few steps with this input
            for _ in range(steps_per_img):
                P.rates[:] = img.flatten().to(net.device).clone()
                net.iterate()

            # Logging
            if img_idx_global % log_every == 0:
                print(f"\n=== Epoch {epoch}, image {img_idx_global}, label={label} ===")
                log_population_stats(net)
                log_compartment_stats(net)
                print(f"Instant P–E corr: {P_E_corr(net):+.3f}")
                print("=========================================\n")

            img_idx_global += 1

    return net


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unsupervised MNIST drive for recurrent E/I network."
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-img", type=int, default=10)
    parser.add_argument("--warm_up_per_img", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--log-every", type=int, default=500)

    args = parser.parse_args()

    _ = train_mnist_unsupervised(
        epochs=args.epochs,
        steps_per_img=args.steps_per_img,
        warm_up_per_img=args.steps_per_img,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        log_every=args.log_every,
    )