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
    W,H,Z = size_e
    Zi = max(1, int(round(frac_i * Z)))
    return [W,H,Zi]

def ema_update(x, new, alpha):
    return (1 - alpha) * x + alpha * new

def class_means_E2(net):
    """Mean over layers for E2 -> (10,)"""
    E2 = net.populations["E2"]
    W,H,Y = E2.size  # (10,1,Y)
    r = E2.rates.view(W,H,Y)
    means = r.mean(dim=(1,2))  # (10,)
    return means

# Running stats for z-scored logits
class ReadoutStats:
    def __init__(self, ncls=10, device="cpu", tau=200):
        self.mu  = torch.zeros(ncls, device=device)
        self.sig = torch.ones(ncls,  device=device)
        self.alpha = 1.0 / (1.0 + tau)

    def update(self, x):
        # x: (10,)
        # standard EMA for mean and std (simple second-moment EMA)
        self.mu  = ema_update(self.mu, x, self.alpha)
        diff     = x - self.mu
        var_ema  = ema_update(self.sig**2, diff**2, self.alpha)
        self.sig = torch.sqrt(var_ema + 1e-6)

    def logits(self, x, T=1.0):
        z = (x - self.mu) / (self.sig + 1e-6)
        return z / T

def margin_reward_from_means(means, y, clip=1.0):
    """Margin = m_y - max_{j!=y} m_j  (EMA means recommended)."""
    m_true = means[y]
    m_other = torch.max(means[torch.arange(means.numel(), device=means.device) != y])
    r = torch.clamp(m_true - m_other, -clip, clip)
    return float(r)

def set_reward_M(net, comps_ids, reward_value):
    """Apply scalar M to selected compartments by id."""
    for cid in comps_ids:
        for p in net.populations.values():
            if cid in p.compartments:
                p.compartments[cid].M = reward_value
                break

# Normalize Hebbian LR by expected rate scale (pre_r0 * post_r0)
def eta_scaled(eta_base, r0_pre, r0_post, eps=1e-6):
    return eta_base / max(r0_pre * r0_post, eps)

# -------------------------
# Build network
# -------------------------
def build_net(device="cuda" if torch.cuda.is_available() else "cpu",
              Z1=4, Y2=8, frac_i=0.28):

    device = torch.device(device)

    # Shapes
    size_P  = [28,28,1]
    size_E1 = [28,28,Z1]
    size_I1 = make_sizes_ei(size_E1, frac_i=frac_i)
    size_E2 = [10,1,Y2]
    size_I2 = make_sizes_ei(size_E2, frac_i=frac_i)

    def active_fun(u):
        s = u[next(iter(u))].clone()
        s[:] = torch.randn_like(s)*5
        for i in u.values():
            s+=i
        return s


    R_E = 70
    R_I = 150
    # Populations (r0 chosen near expected operating ranges)
    pops = {}
    pops["P"]  = population_parameters("P",  size=size_P,  tau=1, rate_inflection=255.0, activation_exponent=1.0,  baseline=0.0, cap=600.0,activation=None)
    pops["E1"] = population_parameters("E1", size=size_E1, tau=3, rate_inflection=R_E*1.5,  activation_exponent=0.85, baseline=0.0, cap=600.0,activation=None)
    pops["I1"] = population_parameters("I1", size=size_I1, tau=2, rate_inflection=R_I*1.5,  activation_exponent=1.5, baseline=0.0, cap=1000.0,activation=None)
    pops["E2"] = population_parameters("E2", size=size_E2, tau=3, rate_inflection=R_E*1.5,  activation_exponent=0.85, baseline=0.0, cap=600.0,activation=None)
    pops["I2"] = population_parameters("I2", size=size_I2, tau=2, rate_inflection=R_I*1.5,  activation_exponent=1.5, baseline=0.0, cap=1000.0,activation=None)

    # Homeostasis / amplitude
    AVG_TAU = 500
    # Hebbian covariance
    AVG_TAU2 = 30
    DELTA_E =  0.1/AVG_TAU*1   # roughly 10% amp gain after AVG_TAU  for rate off by factor 2# excitatory amplitude learning (positive)
    DELTA_I =  -0e-5   # inhibitory amplitude learning (negative)
    RHO     =  0
    KAPPAT   = 1-1./0.4 # target a CV of 0.4
    KAPPAS   = 1-1./0.4
    GAMMAS = DELTA_E*0.05
    GAMMAT = DELTA_E*0.01
    # Learning-rate base values (will be scaled by r0_pre*r0_post)
    BASE = 0.05*0.1 # maximal 0.05*1/k <rj>/<ri> (but shouldn't be smaller than this than a factor of 1/TAU_AVG)
    LR_EE = 2*BASE
    LR_EI = 1*BASE
    LR_IE = 4*BASE
    LR_II = 1*BASE
    BETA_E  = BASE*0.01
    BETA_I  = BASE*0.1
    LEARN1 = 1.01
    LEARN2 = 1.

    ALPHA = BASE/R_E/R_E*0.1
    # Frequency band (optional)
    bands_e1 = {}
    bands_e1["theta"] = {"period": 11, "tau": 16, "alpha": ALPHA*LEARN1}
    bands_e1["gamma"] = {"period": 4, "tau": 6, "alpha": ALPHA*LEARN1}
    bands_e1["quick"] = {"period": 3, "tau": 4, "alpha": ALPHA*LEARN1}
    bands_e2 = {}
    bands_e2["theta"] = {"period": 11, "tau": 16, "alpha": ALPHA*LEARN2}
    bands_e2["gamma"] = {"period": 4, "tau": 6, "alpha": ALPHA*LEARN2}
    bands_e2["quick"] = {"period": 3, "tau": 4, "alpha": ALPHA*LEARN2}
    bands_i = {}

    # Pull r0s for eta scaling
    r0P  = pops["P"]["r0"]
    r0E1 = pops["E1"]["r0"]; r0I1 = pops["I1"]["r0"]
    r0E2 = pops["E2"]["r0"]; r0I2 = pops["I2"]["r0"]

    comps = {}

    # ----------- P -> E1 (local FF; fixed weights; amplitude allowed) ------------
    comps["P_E1"] = compartment_parameters(
        id="P_E1", source="P", target="E1",
        ellipse=[3,3], tsyn=16,
        A=1.8, A0=1.8,                 # slightly conservative due to strong input
        eta=eta_scaled(LR_EE, R_E, R_E)*LEARN1*0.1, beta=0.0,
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=0,
        rin=1.0, rout=1.0, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E*0, rate_target=R_E,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )

    # ----------- E1/I1 recurrent (local) ------------
    comps["E1_E1"] = compartment_parameters(
        id="E1_E1", source="E1", target="E1",
        ellipse=[4,4], tsyn=18,
        A=5.5, A0=5.5,
        eta=eta_scaled(LR_EE, R_E, R_E)*LEARN1*1., beta=BETA_E*LEARN1*0,
        bands=bands_e1, rho=RHO, tau=AVG_TAU,nu=0,
        rin=1, rout=1, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E*LEARN1, rate_target=R_E,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )
    comps["E1_I1"] = compartment_parameters(
        id="E1_I1", source="E1", target="I1",
        ellipse=[4,4], tsyn=18,
        A=5.5, A0=5.5,
        eta=eta_scaled(LR_EI, R_E, R_I)*LEARN1, beta=BETA_E*LEARN1,
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=0,
        rin=1, rout=1, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E*LEARN1*0.1, rate_target=R_I,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )
    comps["I1_E1"] = compartment_parameters(
        id="I1_E1", source="I1", target="E1",
        ellipse=[3,3], tsyn=18,
        A=-5.5, A0=5.5,
        eta=eta_scaled(LR_IE, R_E, R_I)*LEARN1*1.0, beta=BETA_I*LEARN1,
        bands=bands_i, rho=GAMMAS*0.1, tau=AVG_TAU,nu=0,
        rin=0, tauin=-AVG_TAU2, rout=3*R_E, tauout=-AVG_TAU2,   # post-error inhibition
        delta=DELTA_I*LEARN1, rate_target=R_E,
        gammas=GAMMAS*LEARN1*0,kappas=KAPPAS,gammat=GAMMAT*LEARN1*0,kappat=KAPPAT
    )
    comps["I1_I1"] = compartment_parameters(
        id="I1_I1", source="I1", target="I1",
        ellipse=[3,3], tsyn=18,
        A=-3.0, A0=3.0,
        eta=eta_scaled(LR_II, R_I, R_I)*LEARN1, beta=BETA_I*LEARN1,
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=0,
        rin=0, tauin=-AVG_TAU2, rout=3*R_I, tauout=-AVG_TAU2,
        delta=DELTA_I*LEARN1, rate_target=R_I,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )

    # ----------- E2/I2 recurrent (dense) ------------
    comps["E2_E2"] = compartment_parameters(
        id="E2_E2", source="E2", target="E2",
        ellipse=[0.5,0.5], tsyn=-30,
        A=4, A0=4,
        eta=eta_scaled(LR_EE, R_E, R_E)*1, beta=BETA_E*0.0,  # start without recurrent E2 plasticity; can turn on later
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=0,
        rin=1, rout=1, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E, rate_target=R_E,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )
    comps["E2_I2"] = compartment_parameters(
        id="E2_I2", source="E2", target="I2",
        ellipse=[50,50], tsyn=-20,
        A=5.5, A0=5.5,
        eta=eta_scaled(LR_EI, R_E, R_I)*1, beta=BETA_E*0,
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=0,
        rin=1.0, rout=1.0, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E*0.1, rate_target=R_I,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )
    comps["I2_E2"] = compartment_parameters(
        id="I2_E2", source="I2", target="E2",
        ellipse=[0.5,0.5], tsyn=-30,
        A=-6.5, A0=6.5,   # a bit stronger to curb saturation
        eta=eta_scaled(LR_IE, R_E, R_I)*1, beta=BETA_I*0.0,
        bands=bands_i, rho=GAMMAS*0.1, tau=AVG_TAU,nu=0,
        rin=0, tauin=-AVG_TAU2, rout=3*R_E, tauout=-AVG_TAU2,
        delta=DELTA_I*0, rate_target=R_E,
        gammas=GAMMAS*0,kappas=KAPPAS,gammat=GAMMAT*0,kappat=KAPPAT
    )
    comps["I2_I2"] = compartment_parameters(
        id="I2_I2", source="I2", target="I2",
        ellipse=[50,50], tsyn=-20,
        A=-3.0, A0=3.0,
        eta=eta_scaled(LR_II, R_I, R_I), beta=BETA_I*0,
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=0,
        rin=0, tauin=-AVG_TAU2, rout=R_I*3, tauout=-AVG_TAU2,
        delta=DELTA_I, rate_target=R_I,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )

    # ----------- Cross-pool EXCITATORY ONLY (reward-gated) ------------
    comps["E1_E2"] = compartment_parameters(
        id="E1_E2", source="E1", target="E2",
        ellipse=[50,50], tsyn=-100,
        A=2, A0=2,  # toned down due to external input
        eta=eta_scaled(LR_EE, R_E, R_E)*0.1, beta=BETA_E*0.0,
        bands=bands_i, rho=RHO, tau=AVG_TAU,nu=eta_scaled(LR_EE, R_E, R_E)*1,
        rin=1, rout=1, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E*0, rate_target=R_E,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )
    comps["E2_E1"] = compartment_parameters(
        id="E2_E1", source="E2", target="E1",
        ellipse=[50,50], tsyn=-5,
        A=0.25, A0=0.25,  # keep small to avoid runaway loops
        eta=eta_scaled(LR_EE, R_E, R_E)*LEARN1*0.0, beta=BETA_E*0.0,
        bands={}, rho=RHO, tau=AVG_TAU,nu=eta_scaled(LR_EE, R_E, R_E)*0.00,
        rin=1, rout=1, tauin=AVG_TAU2, tauout=AVG_TAU2,
        delta=DELTA_E*0, rate_target=R_E,
        gammas=0,kappas=KAPPAS,gammat=0,kappat=KAPPAT
    )

    net = Network(device, pops, comps)
    return net

# -------------------------
# MNIST training (instantaneous z-logits + margin reward)
# -------------------------
def train_mnist(
    epochs=1,
    steps_per_img=10,
    device=None,
    batch_size=1,
    seed=123,
    reward_T=1.0,
    stats_tau=200,
    log_every=500,
    # new scheduling knobs
    global_warmup_images=12_000,   # no reward for first N images
    reward_ramp_images=2_000,     # then ramp reward scale to 1.0 over this many images
    per_image_settle=2,           # zero-reward settle steps at image onset
    reward_delay_steps=2,         # extra zero-reward steps per image before reward kicks in
    ramp_type="linear"            # "linear" or "cosine"
):
    import torchvision as tv
    import torchvision.transforms as T
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([T.ToTensor()])
    trainset = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
    )

    net   = build_net(device=device)
    stats = ReadoutStats(ncls=10, device=net.device, tau=stats_tau)

    # Reward routing: start with feed-forward only
    reward_comps = ["E1_E2","E2_E1"]  # add "E2_E1" later if you want
    #reward_comps = []

    correct = 0
    total   = 0
    means_ema   = torch.zeros(10, device=net.device)
    alpha_means = 0.4
    conf = torch.zeros(10, 10, dtype=torch.long)

    def reward_scale_for_image(img_idx: int) -> float:
        """0 during global warmup; then ramp to 1 over reward_ramp_images."""
        if img_idx < global_warmup_images:
            return 0
        t = img_idx - global_warmup_images
        if reward_ramp_images <= 0:
            return 0
        x = max(0.0, min(1.0, t / reward_ramp_images))
        if ramp_type == "cosine":
            # smooth start: 0 -> 1
            return 0.5 * (1 - math.cos(math.pi * x))
        return x  # linear

    img_idx_global = 0

    for epoch in range(epochs):
        for (imgs, labels) in loader:
            img   = imgs[0, 0] * 255.0
            label = int(labels[0].item())

            # Input population
            P = net.populations["P"]
            P.rates = img.flatten().to(net.device).clone()

            # --- per-image settle with zero reward ---
            # set to default learning value of 0
            set_reward_M(net, reward_comps, 0)
            for _ in range(per_image_settle):
                P.rates = img.flatten().to(net.device).clone()
                net.iterate()

            # --- per-image steps with delayed reward & ramped scale ---
            for step_idx in range(steps_per_img):
                # Snapshot readout
                means_now = class_means_E2(net)
                means_now = torch.nan_to_num(means_now, nan=0.0, posinf=1e6, neginf=0.0)

                stats.update(means_now)
                zlogits_now = stats.logits(means_now, T=reward_T)

                # Same-snapshot pred
                pred_now = int(torch.argmax(zlogits_now).item())

                # Margin on same snapshot
                mask = torch.ones(10, dtype=torch.bool, device=net.device)
                mask[label] = False
                margin_now = (zlogits_now[label] - torch.max(zlogits_now[mask])).item()

                # Schedule: neutral until (a) per-image delay passed AND (b) global warmup/ramp allows it
                if step_idx < reward_delay_steps:
                    reward_now = 0
                else:
                    #base = max(-1.0, min(1.0, margin_now))
                    base = max(0, min(1.0, margin_now))
                    scale = reward_scale_for_image(img_idx_global)
                    reward_now = float(scale * base)

                '''
                # Optional consistency warning (same snapshot)
                if (pred_now != label) and (reward_now > 0.0):
                    print(f"[warn-step] positive reward while mispredicted "
                          f"(same snapshot): pred={pred_now} label={label} "
                          f"margin={margin_now:+.3f} scale={reward_scale_for_image(img_idx_global):.3f}")
                '''

                # Apply reward and step
                set_reward_M(net, reward_comps, reward_now)
                net.iterate()

            # ---- per-image evaluation (after last iterate) ----
            with torch.no_grad():
                means_fin  = class_means_E2(net)
                means_fin  = torch.nan_to_num(means_fin, nan=0.0, posinf=1e6, neginf=0.0)
                zlogits_ev = stats.logits(means_fin, T=reward_T)
                pred_fin   = int(torch.argmax(zlogits_ev).item())
                correct   += int(pred_fin == label)
                total     += 1
                conf[pred_fin, label] += 1

            # Logging EMA (pretty only)
            means_ema = ema_update(means_ema, means_fin, alpha_means)
            probs_ev  = torch.softmax(zlogits_ev, dim=0)

            if total % log_every == 0:
                acc = 100.0 * correct / total
                means_str = ", ".join([f"{m:.1f}" for m in means_ema.tolist()])
                probs_str = ", ".join([f"{p:.2f}" for p in probs_ev.tolist()])
                print(f"[{epoch}:{total}] acc={acc:.2f}%  pred={pred_fin}  label={label}  "
                      f"reward_scale={reward_scale_for_image(img_idx_global):.3f}  "
                      f"reward margin={margin_now:.3f}")
                print(f"    E2 means(EMA): [{means_str}]")
                print(f"    E2 probs(z):   [{probs_str}]")
                print("\n--- Network Activity Summary ---")
                for pid, pop in net.populations.items():
                    r = pop.rates.detach().cpu()

                    mean_r = r.mean().item()
                    std_r  = r.std(unbiased=False).item()  # population variance
                    r_avg = 0
                    CV_t = 0
                    if(len(pop.compartments)>0):
                        c = pop.compartments[next(iter(pop.compartments))]
                        ra = c.rate_average.detach().cpu()
                        r_avg = ra.mean().item()
                        r_std = ra.std(unbiased=False).item()
                        CV_t = r_std/r_avg


                    print(f"Population {pid:>4s} | mean rate = {mean_r:7.3f}  |  CV_s = {(std_r/mean_r):7.3f}  | r_avg = {r_avg:7.3f}  | CV_t = {CV_t:7.3f}")
                print("--------------------------------\n")
                print("\n--- Network Weight Summary ---")
                for pid, pop in net.populations.items():
                    for cid, comp in pop.compartments.items():
                        a = comp.a.detach().cpu()
                        w = comp.w.detach().cpu()
                        

                        mean_a = a.mean().item()
                        std_a = a.std(unbiased=False).item()
                        std_w = w.std(unbiased=False).item()
                        r_avg = ra.mean().item()
                        r_std = ra.std(unbiased=False).item()

                        print(f"{comp.sourceid+"-"+comp.targetid:>8s} | mean amp = {mean_a:7.3f}  | std amp = {std_a:7.4f}  |  std w = {(std_w):7.5f}")
                print("--------------------------------\n")

            img_idx_global += 1  # advance image counter

    acc = 100.0 * correct / max(total, 1)
    print(f"Final train accuracy (instant z-logits): {acc:.2f}%")
    print("Confusion (rows=pred, cols=true):")
    print(conf.cpu().numpy())
    return net

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Train rate-based MNIST with instantaneous z-logit margin reward.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-img", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--reward-T", type=float, default=1.0)
    parser.add_argument("--stats-tau", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--global-warmup-images", type=int, default=2000)
    parser.add_argument("--reward-ramp-images", type=int, default=8000)
    parser.add_argument("--per-image-settle", type=int, default=1)
    parser.add_argument("--reward-delay-steps", type=int, default=1)
    parser.add_argument("--ramp-type", type=str, choices=["linear", "cosine"], default="linear")

    args = parser.parse_args()

    _ = train_mnist(
        epochs=args.epochs,
        steps_per_img=args.steps_per_img,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        reward_T=args.reward_T,
        stats_tau=args.stats_tau,
        log_every=args.log_every,
        global_warmup_images=args.global_warmup_images,
        reward_ramp_images=args.reward_ramp_images,
        per_image_settle=args.per_image_settle,
        reward_delay_steps=args.reward_delay_steps,
        ramp_type=args.ramp_type,
    )
