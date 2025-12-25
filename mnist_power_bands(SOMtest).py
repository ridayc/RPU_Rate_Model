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
    SOM,row_sum
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
    size_S = [28, 28, 1]

    # --- Population definitions ---
    R_E = 1.
    R_S = 1.5
    R_I = 5.

    baseline = 0.02


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
        tau=4,
        rate_inflection=R_E * 1.5,
        activation_exponent=0.55,
        baseline=baseline,
        cap=60.0,
        activation=None,
    )
    pops["I"] = population_parameters(
        "I",
        size=size_I,
        tau=2,
        rate_inflection=R_I * 1.5,
        activation_exponent=1.5,
        baseline=baseline,
        cap=360.0,
        activation=None,
    )

    pops["S"] = population_parameters(
        "S",
        size=size_S,
        tau=4,
        rate_inflection=R_S * 1,
        activation_exponent=1.0,
        baseline=baseline,
        cap=120.0,
        activation=None,
    )

    # --- Global timescales / learning rates ---
    AVG_TAU  = 900             # long-term averages for amplitude, etc.
    TAU_COV = 900              # long-term averages for covariance hebb
    TAU_BCM = 20               # short term averages for bcm facilitation/supression
    TAUW = 100             # time scale for plasisticity smoothing
    TAU_HOMEO_E = 900        # slow time scale for homeostatic learning of E types
    TAU_HOMEO_I = TAU_HOMEO_E*4      # slow time scale for homeostatic learning of E types
    TAU_SLOW = TAU_HOMEO_I
    TAUL = 3000                # long-term average for LTP/LTD balancing
    META_E = 0.001/TAUL*0         # learning rate for meta plasticitcy of LTP/LTD
    META_I = 0.001/TAUL*0        
    REG_E = 0.1/TAUL         # learning rate for meta plasticitcy of LTP/LTD
    REG_I = 0.1/TAUL         # learning rate for meta plasticitcy of learning speed beta of regularizer beta*(1/k-wij)
    DELTA_E  = 0.01 / TAU_HOMEO_E  # excitatory amp learning
    DELTA_I = 0.003 / TAU_HOMEO_I       # inhibitory amp learning (off for now)
    DELTA_S = 0.003 / TAU_HOMEO_I
    RHO      = 0.00 / TAU_HOMEO_I 
    RHO_E = DELTA_E*0.00
    RHO_I = DELTA_I*0.00

    # Base Hebbian LR scale
    BASE  = 0.05 * 0.001*(1+TAUW)  # Base learning rate for synaptic plasticity. This scales with AVG_TAU2 as the smoothing averages over the same timescale
    LR_EE = 1 * BASE
    LR_EI = 0.2 * BASE
    LR_ES = 0.05 * BASE
    LR_IE = 0.1 * BASE
    LR_II = 0.05 * BASE
    LR_IS = 0.1 * BASE
    LR_SE = 0.5 * BASE
    LR_SI = -0 * BASE
    LR_SS = -0.005 * BASE
    BETA_EE = 1.*LR_EE
    BETA_EI = 1.*LR_EI
    BETA_ES = 1.*LR_ES
    BETA_IE = 1.*LR_IE
    BETA_II = 1.*LR_II
    BETA_IS = 0*LR_IS
    BETA_SE = 1.*LR_SE
    BETA_SI = 0*LR_SI
    BETA_SS = 0*LR_SS

    KAPPA_EE = 0.31
    KAPPA_EI = 0.31
    KAPPA_ES = 0.2
    KAPPA_IE = 0.31
    KAPPA_II = 0.31
    KAPPA_IS = 0.2
    KAPPA_SE = 0.25
    KAPPA_SI = 0.25
    KAPPA_SS = 0.25
    # fraction of 1/k (the mean synapse weight) of the compartment synapses must be below to be counted as silent.
    THETA_R = 0.5

    BN_PE = 2
    BN_EE = 0.5
    BN_EI = 0.5
    BN_ES = 0
    BN_IE = 1
    BN_II = 1
    BN_IS = 0
    BN_SE = 0
    BN_SI = 0
    BN_SS = 0
    BP_PE = BN_PE
    BP_EE = BN_EE
    BP_EI = BN_EI
    BP_ES = 0.5
    BP_IE = BN_IE
    BP_II = BN_II
    BP_IS = 0.2
    BP_SE = 0.25
    BP_SI = 0.25
    BP_SS = 0.25
    # spread of synapse rfs
    RAD_E = 6
    RAD_I = 3
    RAD_S = 4
    AEE = 11

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
    taup = TAU_SLOW
    theta = {"f":[0.2,0.35],"s":[0.2,0.6]}
    eta_out = {"f":[1.,2.],"s":[2.,1.]}
    eta_in = {"f":[0.3,0.1],"s":[0.1,0.3]}
    for i in ["in","out"]:
        E_E_band["synapse"][i]["tau"] = cp.copy(freq)
        E_E_band["synapse"][i]["taup"] = taup
        E_E_band["synapse"][i]["theta"] = cp.deepcopy(theta)
    E_E_band["synapse"]["in"]["eta"] = cp.deepcopy(eta_in)
    E_E_band["synapse"]["out"]["eta"] = cp.deepcopy(eta_out)

    # frequency power bands for amplitude learning (I-I)
    I_I_band = {}
    I_I_band["amplitude"] = {}
    freq = {"f":7,"m":20,"s":60}
    taup = TAU_SLOW
    ETAII = DELTA_E*0.5
    eta = {"f":[2.*ETAII,4.*ETAII],"s":[2.*ETAII,1.*ETAII]}
    theta = {"f":[0.3,0.6],"s":[0.2,0.5]}
    I_I_band["amplitude"]["target"] = "I_I"
    I_I_band["amplitude"]["tau"] = cp.copy(freq)
    I_I_band["amplitude"]["taup"] = taup
    I_I_band["amplitude"]["theta"] = cp.deepcopy(theta)
    I_I_band["amplitude"]["eta"] = cp.deepcopy(eta)

    frequencies = {}
    frequencies["theta"] = {"period": 11, "tau": 16, "alpha": LR_EE/R_E/R_E/3*0}
    #frequencies["gamma"] = {"period": 4, "tau": 6, "alpha": LR_EE/R_E/R_E/3*1}
    #frequencies["quick"] = {"period": 3, "tau": 4, "alpha": LR_EE/R_E/R_E/3*1}

    # --- Compartments dict ---
    comps = {}

    # 1) P -> E: feedforward, fixed-ish
    comps["P_E"] = compartment_parameters(
        id="P_E",
        source="P",
        target="E",
        ellipse=[3.5, 3.5],
        tsyn=50,
        A=0.01,
        A0=0.01,
        eta=LR_EE/50./R_E/R_E*0.1,
        nu=0.0,
        beta=BETA_EE,
        bn=BN_PE,
        bp=BP_PE,
        an=1.,
        ap=0.5,
        taul = TAUL,
        etal = META_E,
        etar = REG_E,
        kappa = KAPPA_EE,
        thetar = THETA_R,
        bands=None,
        rho=0,
        tau=TAU_HOMEO_E,
        tauw=TAUW,
        taug = TAU_SLOW,
        thetaz = 0.05,
        z_value = 1./3,
        zeta = DELTA_E*0.1,
        ratio = "ueff",
        rin=1.0,
        rout=1.0,
        tauin=TAU_COV,
        tauout=-TAU_COV,
        delta=0.0,              # no amplitude learning on feedforward for now
        rate_target=R_E,
        eps=2.0,
        stype="",               # not used for E/PV statistics
    )

    # 2) E -> E: recurrent excitatory, tracked by PV (stype="E")
    comps["E_E"] = compartment_parameters(
        id="E_E",
        source="E",
        target="E",
        ellipse=[RAD_E, RAD_E],
        tsyn=50,
        A=25,
        A0=25,
        eta=LR_EE/R_E/R_E/R_E,
        alpha= 0.,
        nu=0.0,
        beta=BETA_EE,
        bn=BN_EE,
        bp=BP_EE,
        an=1.,
        ap=0.5,
        taul = TAUL,
        etal = META_E,
        etar = REG_E,
        kappa = KAPPA_EE,
        thetar = THETA_R,
        bands=E_E_band,
        freq=frequencies,
        rho=RHO_E,
        tau=TAU_HOMEO_E,
        taug=TAU_SLOW,
        tauw=TAUW,
        rin=1,
        rout=1,
        tauin=TAU_COV,
        tauout=-TAU_COV,
        taub=TAU_HOMEO_E,
        zeta=0,
        z_value = 0,
        ratio = "E/I",
        delta=DELTA_E,
        rate_target=R_E,
        eps=1.0,
        stype="",
        stat = True,
        power={
            "tauf": TAU_BCM,   # fast mixing timescale
            "taus": TAU_SLOW,      # slow mixing timescale
        }
    )

    # 3) E -> I: excitatory drive to inhibitory
    comps["E_I"] = compartment_parameters(
        id="E_I",
        source="E",
        target="I",
        ellipse=[RAD_I, RAD_I],
        tsyn=50,
        A=1.5,
        A0=1.5,
        eta=LR_EI/R_E/R_I/R_I,
        nu=0.0,
        beta=BETA_EI,
        bn=BN_EI,
        bp=BP_EI,
        an=1,
        ap=0.5,
        taul = TAUL,
        etal = META_E,
        etar = REG_E,
        kappa = KAPPA_EI,
        thetar = THETA_R,
        bands=None,
        rho=RHO_E*0.1,
        tau=TAU_HOMEO_E,
        tauw=TAUW,
        taug=TAU_SLOW,
        rin=1,
        rout=1,
        tauin=TAU_COV,
        tauout=-TAU_COV,
        taub=TAU_BCM,
        noise=0,
        cv=0,
        zeta=DELTA_E*0,
        z_value = 0.4,
        ratio = "E/I",
        rq = 0.05*0,
        rt=5.,
        delta=DELTA_E*0.4,   # small amplitude plasticity
        rate_target=R_I,
        eps=1.0,
        stype="",              # not tracked by PV statistics
    )

    # 4) E -> S: excitatory drive to SOM
    comps["E_S"] = compartment_parameters(
        id="E_S",
        source="E",
        target="S",
        ellipse=[RAD_S, RAD_S],
        tsyn=50,
        A=2*R_E,
        A0=2*R_E,
        eta=LR_ES/R_E/R_S/R_S,
        nu=0.0,
        beta=BETA_ES,
        bn=BN_ES,
        bp=BP_ES,
        taul = TAUL,
        etal = META_E,
        kappa = KAPPA_ES,
        thetar = THETA_R,
        bands=None,
        rho=RHO_E*0.00,
        tau=TAU_HOMEO_E,
        taug=TAU_SLOW,
        tauw=TAUW,
        rin=1,
        rout=1,
        tauin=TAU_COV,
        tauout=-TAU_COV,
        taub=TAU_BCM,
        zeta=0,
        z_value = 0,
        ratio = "E/I",
        delta=DELTA_E*0.1,   # small amplitude plasticity
        rate_target=R_S,
        eps=1.0,
        stype="",              # not tracked by PV statistics
        SOM=SOM(som_type="post",c=AEE,eta_k=DELTA_I),
    )

    # 5) I -> E: inhibitory PV-like compartment (stype="PV")
    comps["I_E"] = compartment_parameters(
        id="I_E",
        source="I",
        target="E",
        ellipse=[RAD_I, RAD_I],
        tsyn=20,
        A=-31.,               # inhibitory
        A0=31.,               # target amplitude (magnitude)
        eta=LR_IE/R_I/R_E,
        nu=0.0,
        beta=BETA_IE,
        bn=BN_IE,
        bp=BP_IE,
        an = 1,
        ap = 0.5,
        taul = TAUL,
        etal = META_I,
        etar = REG_I,
        kappa = KAPPA_IE,
        thetar = THETA_R,
        bands=None,
        rho=RHO_I,              # you might later add rho for loga regularization
        tau=TAU_HOMEO_E,
        taug=TAU_SLOW,
        tauw=TAUW,
        taub=TAU_BCM,
        rin=0,
        tauin=TAU_COV,
        rout=1,
        tauout=TAU_COV,
        zeta=-DELTA_E*0.2,
        zeta2=-DELTA_E*0.4,
        z_value = 0.4,
        ratio = "E/I",
        rq = 0.00,
        rt=0,
        delta=-DELTA_E*0,        # inhibitory amplitude learning (off now)
        rate_target=R_E,
        eps=1.0,
        stype="",
    )

    # 6) I -> I: inhibitory recurrent, not PV-tracked
    comps["I_I"] = compartment_parameters(
        id="I_I",
        source="I",
        target="I",
        ellipse=[RAD_I, RAD_I],
        tsyn=20,
        A=-0.5,
        A0=0.5,
        eta=-LR_II/R_I/R_I,
        nu=0.0,
        beta=BETA_II,
        bn=BN_II,
        bp=BP_II,
        an = 1,
        ap = 0.5,
        taul = TAUL,
        etal = META_I,
        etar = REG_I,
        kappa = KAPPA_II,
        thetar = THETA_R,
        bands=I_I_band,
        rho=RHO_I,
        tauw=TAUW,
        tau=TAU_SLOW,
        taub=TAU_SLOW,
        taug=TAU_SLOW,
        rin=0,
        tauin=TAU_COV,
        rout= 1,
        tauout=TAU_COV,
        zeta=DELTA_E*1,
        z_value = 0.4,
        ratio = "corr",
        rq = 0,
        rt=5.,
        delta=-DELTA_E*0,
        rate_target=R_I,
        eps=1.0,
        stype="",
    )

    # 7) I -> S: inhibitory PV-like compartment on S (stype="PV")
    comps["I_S"] = compartment_parameters(
        id="I_S",
        source="I",
        target="S",
        ellipse=[RAD_I, RAD_I],
        tsyn=20,
        A=-3.5,               # inhibitory
        A0=3.5,               # target amplitude (magnitude)
        eta=LR_IS/R_S/R_I/R_S,
        nu=0.0,
        beta=BETA_IS,
        kappa = KAPPA_IS,
        thetar = THETA_R,
        bands=None,
        rho=RHO_I,              # you might later add rho for loga regularization
        tau=TAU_SLOW,
        taug=TAU_SLOW,
        tauw=TAUW,
        rin=0.0,
        tauin=TAU_COV,
        rout=1.5*R_S,
        tauout=-TAU_COV,
        zeta=DELTA_I*0.2,
        z_value = 0.33,
        ratio = "E/I",
        delta=-DELTA_I*0.1,        # inhibitory amplitude learning (off now)
        rate_target=R_S,
        eps=1.0,
        stype="",
    )

    # 8) S -> E: inhibitory SOM-like compartment (stype="SOM")
    comps["S_E"] = compartment_parameters(
        id="S_E",
        source="S",
        target="E",
        ellipse=[RAD_S, RAD_S],
        tsyn=25,
        A=-2.5,               # inhibitory
        A0=2.5,               # target amplitude (magnitude)
        eta=LR_SE/R_S,
        nu=0.0,
        beta=BETA_SE,
        bn=BN_SE,
        bp=BP_SE,
        an=1,
        ap=1,
        taul = TAUL,
        etal = META_I,
        kappa = KAPPA_SE,
        thetar = THETA_R,
        bands=None,
        rho=RHO_I*0,              # you might later add rho for loga regularization
        tau=TAU_SLOW,
        taug=TAU_SLOW,
        tauw=TAUW,
        rin=0,
        tauin=TAU_SLOW,
        rout=1,
        tauout=TAU_SLOW,
        zeta=DELTA_S*8,
        z_value = 0.2,
        ratio = "Ieff",
        delta=0,        # inhibitory amplitude learning (off now)
        rate_target=R_E,
        eps=0,
        stype="",
        SOM=SOM(som_type="pre",c=AEE,eta_k=DELTA_I),
    )

    # 8) S -> I: inhibitory SOM-like compartment (stype="SOM")
    comps["S_I"] = compartment_parameters(
        id="S_I",
        source="S",
        target="I",
        ellipse=[RAD_I, RAD_I],
        tsyn=25,
        A=-1,               # inhibitory
        A0=1,               # target amplitude (magnitude)
        eta=LR_SI/R_S/R_I,
        nu=0.0,
        beta=BETA_SI,
        kappa = KAPPA_SI,
        thetar = THETA_R,
        bands=None,
        rho=RHO_I,              # you might later add rho for loga regularization
        tau=TAU_SLOW,
        taug=TAU_SLOW,
        tauw=TAUW,
        rin=0.0,
        tauin=TAU_COV,
        rout=R_I*1.5,
        zeta=DELTA_S,
        z_value = 0.1,
        ratio = "Ieff",
        tauout=TAU_COV,
        delta=0,        # inhibitory amplitude learning (off now)
        rate_target=R_I,
        eps=0.0,
        stype="",
    )

    # 8) S -> I: inhibitory SOM-like compartment (stype="SOM")
    comps["S_S"] = compartment_parameters(
        id="S_S",
        source="S",
        target="S",
        ellipse=[RAD_S, RAD_S],
        tsyn=25,
        A=-0.5,               # inhibitory
        A0=0.5,               # target amplitude (magnitude)
        eta=LR_SS/R_S/R_S/R_S,
        nu=0.0,
        beta=BETA_SS,
        kappa = KAPPA_SS,
        thetar = THETA_R,
        bands=None,
        rho=RHO_I,              # you might later add rho for loga regularization
        tau=TAU_SLOW,
        taug=TAU_SLOW,
        tauw=TAUW,
        rin=R_S*1.5,
        tauin=-TAU_COV,
        rout=R_S*0.5,
        zeta=DELTA_S,
        z_value = 0.1,
        ratio = "Ieff",
        tauout=TAU_COV,
        delta=0,        # inhibitory amplitude learning (off now)
        rate_target=R_S,
        eps=0.0,
        stype="",
    )

    del pops["S"];del comps["E_S"];del comps["I_S"];del comps["S_E"];del comps["S_I"];del comps["S_S"]
    #del comps["S_I"];del comps["S_S"]; del comps["I_S"]
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
            m = comp.rit_slow.detach().cpu()

            def safe_mean(x):
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                return float(x.mean().item())

            print(
                f"[{pid}:{cid:>4s}] "
                f"CVt_med={torch.median(CVt/(m+1e-8)):6.3f}  "
                f"CVs_med={torch.median(CVs):6.3f}  "
                f"C_med={torch.median(C):7.4f}  "
                f"C2_mean={safe_mean(C2):7.4f}"
            )

    print("----------------------------------------\n")

    print("\n--- Compartment Power / SOM quantiles ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            if "synapse" in comp.rate_band:

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
            if "amplitude" in comp.rate_band:

                Pf = comp.rate_band["amplitude"]["p"]["f"].detach().cpu()
                Pm = comp.rate_band["amplitude"]["p"]["m"].detach().cpu()
                Ps = comp.rate_band["amplitude"]["p"]["s"].detach().cpu()
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
            if comp.SOM!=None:
                print(
                    f"[{pid}:{cid:>4s}] "
                    f"k={torch.median(comp.SOM.k.mean()):6.3f}  "
                    f"q={torch.median(comp.SOM.quantile.mean()):6.3f}  "
                )

    print("----------------------------------------\n")

    print("\n--- Network Weight Summary ---")
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            a = comp.a.detach().cpu()
            w = comp.w.detach().cpu()
            wind = comp.w_ind[0,:].detach().cpu()
            # assuming numerator/denominator exist in your Compartment implementation
            ES = comp.numerator.detach().cpu()
            IS = comp.denominator.detach().cpu()
            if(comp.rq>0):
                G=torch.mean(comp.rate_q.detach().cpu())
            elif(comp.ratio=="corr" or comp.ratio=="NMC"):
                G = torch.mean(comp.numerator)
            else:
                G = torch.mean(ES/(IS+1e-8))
            rat = math.exp(-comp.dM.detach().cpu().median())
            Neff = (1/comp.k*row_sum((w<comp.thetar).float(),wind)).mean()
            bfact = math.exp(comp.dN.detach().cpu().median())
            
            mean_a = a.mean().item()
            std_a = a.std(unbiased=False).item()
            m_w = w.mean()
            std_w = w.std(unbiased=False).item()

            print(f"{(comp.sourceid+'-'+comp.targetid):>8s} | A_m = {mean_a:7.3f}  | A_s = {std_a:7.4f}  |  CV(w) = {(std_w/m_w):7.3f} | an/ap = {(rat):7.5f} | N = {(Neff):7.5f} | b = {(bfact):7.5f} |  I-E = {(G):7.3f}")
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
    parser.add_argument("--steps-per-img", type=int, default=10)
    parser.add_argument("--warm_up_per_img", type=int, default=5)
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