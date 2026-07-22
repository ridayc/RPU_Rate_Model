import math
import copy as cp
import os
import torch

from network import (
    Network,
    population_parameters,
    compartment_parameters,
    SST
)


# ============================================================
# Topology configuration
# Define which compartments are active. Populations are inferred
# automatically from active compartments so no manual deletion needed.
# ============================================================
ACTIVE_COMPARTMENTS = [
    "P_E",
    "E_E",
    "E_I",
    "I_E",
    "I_I",
    #"E_S","S_E","I_S","S_I", # SOM pathway (disabled). No SST->SST connectivity
    # "E_S", "I_S", "S_E", "S_I", "S_S",  # SOM pathway (disabled)
]


# ============================================================
# Size helpers
# ============================================================

def make_size_i(size_e, frac_i=0.28):
    """Inhibitory population size: same W,H as E, ~frac_i fraction of Z layers."""
    W, H, Z = size_e
    return [W, H, max(1, int(round(frac_i * Z)))]


# ============================================================
# Learning parameter block
# All global timescales and learning rates in one place.
# Change SCALE to uniformly slow/speed up all learning relative
# to network dynamics. Individual rates are expressed as
# dimensionless ratios times BASE so their meaning is preserved
# under rescaling.
# ============================================================

def make_learning_params(scale=0.1):
    """
    Build and return the full learning parameter dict.

    Args:
        scale:  Global learning rate rescaling factor. Slows/speeds all
                learning uniformly while preserving timescale ratios.
                Note: log-domain accumulators (loga, dN, dM) scale
                linearly with this factor — see float64 note in network.py.
    """
    p = {}

    # --- Global scale ---
    p["SCALE"] = scale
    p["SLOW"] = 0.1
    p["FAST"] = 0.1
    slow = p["SLOW"]
    cv = 2.5
    cv = cv**3
    frac = 0.05

    # weight quantile target
    kq = 0.2

    # --- Timescales (in simulation steps) ---
    # All tau values are converted to smoothing constants 1/(1+tau) inside
    # compartment_parameters, so these are expressed in natural step units.
    p["TAU_CALC"]     = 1000
    p["TAU_SYN"]      = 100/scale
    p["AVG_TAU"]      = 900/scale/slow             # long-term average for amplitudes/covariance
    
    p["TAU_BCM"]      = 20/slow              # short-term BCM facilitation/suppression
    p["TAUW"]         = 100             # plasticity smoothing (weight update EMA)
    p["TAU_HOMEO_E"]  = 900/scale/slow             # homeostatic timescale for E populations
    p["TAU_HOMEO_I"]  = 90/scale        # homeostatic timescale for I populations (slower)
    p["TAU_HOMEO_S"]  = p["TAU_HOMEO_I"]         # homeostatic timescale for I populations (slower)
    p["TAU_SLOW"]     = p["TAU_HOMEO_I"]
    p["TAUL_E"]       = p["TAU_SYN"]/p["FAST"]            # LTP/LTD balance averaging window
    p["TAUL_I"]       = p["TAUL_E"]
    p["TAUL_S"]       = p["TAUL_I"]
    p["TAUL_P"]       = p["TAUL_E"]

    # --- Amplitude learning rates ---
    # Expressed as total fractional change per TAU_HOMEO steps, then
    # divided by tau to get per-step rate, then scaled by SCALE.
    p["DELTA_E"]  = 0.1  / p["TAU_HOMEO_E"]   # E amplitude homeostasis
    p["DELTA_I"]  = 0.1 / p["TAU_HOMEO_I"]   # I amplitude homeostasis
    p["DELTA_S"]  = 0.1 / p["TAU_HOMEO_S"]   # SST amplitude homeostasis
    p["ZETA_P"] = 0.1  / p["TAUL_P"]
    p["ZETA_E"] = 0.1  / p["TAUL_E"]
    p["ZETA_I"] = 0.1  / p["TAUL_I"]
    p["ZETA_S"] = 0.1  / p["TAUL_S"]
    p["RHO_E"]    = p["DELTA_E"]*0.01*0                                # E amplitude targeting (to prevent very slow drift)
    p["RHO_I"]    = p["DELTA_I"]*0.005*0                                 # I amplitude targeting

    # --- Meta-plasticity rates ---
    '''
    # LTP/LTD balance learning (an/ap) (dM)
    p["META_P"]  = 0.01  / p["TAUL_P"] *0
    p["META_E"]  = 0.1  / p["TAUL_E"] * 0 
    p["META_I"]  = 0.1   / p["TAUL_I"] * 0
    p["META_S"]  = 0.1   / p["TAUL_S"] * 0
    '''
    p["META_P"]  = 0.1  / p["TAUL_P"]
    p["META_EE"]  = 0.1  / p["TAUL_E"]
    p["META_EI"]  = 0.1  / p["TAUL_E"]
    p["META_IE"]  = 0.1  / p["TAUL_I"]
    p["META_II"]  = 0.1  / p["TAUL_I"]
    p["META_S"]  = 1.5

    # --- Hebbian (covariance) base learning rate ---
    # BASE is normalized so that a typical weight change per step is ~BASE
    # for neurons firing near r0. Individual connection rates are multiples
    # of BASE, reflecting relative plasticity priorities.
    # Since excitatory and inhibitory synapses are "normalized" towards a standard fractional increment per step
    # the ratio of E and I synaptic learning directly controls weight selectivity of neurons.
    # The balance is to have strong E-E learning while making sure that I-E and E-I connectivity can catch up to stabilize selectivity.
    # If E-E is too strong (relatively) learning becomes locked in and subgroups of "burnt in" connectivity form -> the system loses plasisticity
    # If E-E is too weak selectivity is reduced because inhibition too quickly erases all established E-E connectivity structure
    BASE = frac / p["TAU_SYN"] * (1 + p["TAUW"])
    p["LR_EE"] =  1.   * BASE
    p["LR_EI"] =  -2.   * BASE
    p["LR_ES"] =  0.05  * BASE
    p["LR_IE"] =  0.2  * BASE
    p["LR_II"] =  0.2  * BASE
    p["LR_IS"] =  0.2   * BASE
    p["LR_SE"] =  0.02   * BASE
    p["LR_SI"] =  0.03   * BASE
    p["LR_SS"] = -0.005 * BASE
    p["LR_PE"] = 0.01   * BASE

    # weight distribution regularizer (dN)
    p["REG_P"]   = 0.01 /kq    # weight distribution regularizer P
    p["REG_E"]   = 0.01 /kq    # weight distribution regularizer E
    p["REG_I"]   = 0.01 /kq    # weight distribution regularizer I
    p["REG_S"]   = 0.01 /kq    # weight distribution regularizer S

    p["TAU_E"]        = 5.
    p["TAU_I"]        = 2.
    p["TAU_COV_OUT"]  = p["TAU_SYN"]*10            # covariance learning average window on the post synaptic side
    p["TAU_COV_IN"]   = p["TAU_SYN"]*10
    p["TAU_ELIG_E"]   = p["TAU_SYN"]*10
    p["TAU_ELIG_I"]   = 10
    p["TCO_PE"]       = p["TAU_E"]*40 
    p["TCI_PE"]       = p["TAU_E"]*20 
    p["TCO_EE"]       = p["TAU_E"]*2
    p["TCI_EE"]       = p["TAU_E"]*1
    p["TCO_EI"]       = p["TAU_I"]*1 
    p["TCI_EI"]       = p["TAU_E"]*1
    p["TCO_ES"]       = p["TAU_COV_OUT"]
    p["TCI_ES"]       = p["TAU_COV_IN"]
    p["TCO_IE"]       = p["TAU_E"]*1
    p["TCI_IE"]       = p["TAU_E"]*-1
    p["TCO_II"]       = p["TAU_I"]*4
    p["TCI_II"]       = p["TAU_I"]*4
    p["TCO_IS"]       = p["TAU_COV_OUT"]
    p["TCI_IS"]       = p["TAU_COV_IN"]
    p["TCO_SE"]       = -p["TAU_COV_OUT"]
    p["TCI_SE"]       = p["TAU_COV_IN"]
    p["TCO_SI"]       = -p["TAU_COV_OUT"]
    p["TCI_SI"]       = p["TAU_COV_IN"]

    # --- Weight distribution regularizer ---
    p["KAPPA_Q"]  = kq   # target quantile for log-normal shape comparison
    p["Q_R"]      = 0.02  # dead synapse quantile offset (broadens distribution slightly)
    p["THETA_R"]  = 0.4   # minimum CV below which regularization strictly decreases

    # --- LTP/LTD asymmetry exponents ---
    # bn/bp control how strongly large weights are penalized during depression/potentiation
    p["BN_PE"] = 0.25;  p["BP_PE"] = 0.25 #0.25,0.25?
    # excitatory exponent likely has to compensate for the rate exponent of the E population (which is also reflected in the I rates)
    p["BN_EE"] = 0.5;  p["BP_EE"] = 0.5
    p["BN_EI"] = 0.5;  p["BP_EI"] = 0.5
    p["BN_ES"] = 0.5;  p["BP_ES"] = 0.5
    p["BN_IE"] = 0.5;  p["BP_IE"] = 0.5
    p["BN_II"] = 0.5;  p["BP_II"] = 0.5
    p["BN_IS"] = 1.0;  p["BP_IS"] = 1.0
    p["BN_SE"] = 0.25;  p["BP_SE"] = 0.25
    p["BN_SI"] = 0.5;  p["BP_SI"] = 0.5
    p["BN_SS"] = 0.0;  p["BP_SS"] = 0.25

    # --- Weight relaxation rates (pull toward uniform 1/k) --- This relaxation parameter is learned/tuned via the weight distribution regularizer
    b0 = 1.
    p["BETA_PE"] = b0
    p["BETA_EE"] = b0
    p["BETA_EI"] = b0
    p["BETA_ES"] = b0
    p["BETA_IE"] = b0
    p["BETA_II"] = b0
    p["BETA_IS"] = b0
    p["BETA_SE"] = b0
    p["BETA_SI"] = b0
    p["BETA_SS"] = b0
    p["BETA0"]   = 1e-4                 # minimum relaxation floor (log scale)

    p["AN_PE"] = 1;  p["AP_PE"] = 1
    p["AN_EE"] = 1;  p["AP_EE"] = 1
    p["AN_EI"] = 1;  p["AP_EI"] = 1
    p["AN_ES"] = 1;  p["AP_ES"] = 1
    p["AN_IE"] = 1;  p["AP_IE"] = 1
    p["AN_II"] = 1;  p["AP_II"] = 1
    p["AN_IS"] = 1;  p["AP_IS"] = 1
    p["AN_SE"] = 1;  p["AP_SE"] = 1
    p["AN_SI"] = 1;  p["AP_SI"] = 1
    p["AN_SS"] = 1;  p["AP_SS"] = 1

    # --- Receptive field radii (population grid units) ---
    p["RAD_E"] = 6
    p["RAD_I"] = 3
    p["RAD_S"] = 4


    return p


# ============================================================
# Population definitions
# ============================================================

def make_populations(size_E, frac_i=0.28):
    """
    Build population parameter dicts for P, E, I, S.

    Args:
        size_E:  [W, H, Z] for the excitatory population.
        frac_i:  Fraction of E depth for inhibitory population.

    Returns:
        Dict of population parameter dicts keyed by population id.
    """
    size_P = [size_E[0], size_E[1], 1]
    size_I = make_size_i(size_E, frac_i)
    size_S = make_size_i(size_E, frac_i*0.5)

    # Characteristic firing rates (inflection point of RePU nonlinearity)
    r0E = 1.0
    r0I = 10.0
    r0S = 4.0

    bias = 0.001

    pops = {}

    # Input population: linear, no learning, driven externally
    pops["P"] = population_parameters(
        "P",
        size=size_P,
        tau=1,
        rate_inflection=255.0,
        activation_exponent=1.0,
        bias=0,
        cap=600.0,
    )

    # Excitatory population: sub-linear activation (exponent < 1)
    pops["E"] = population_parameters(
        "E",
        size=size_E,
        tau=5,
        rate_inflection=r0E,
        activation_exponent=2.,
        bias=bias,
        cap=600.0,
    )

    # Inhibitory (PV-like) population: super-linear activation for gain control
    pops["I"] = population_parameters(
        "I",
        size=size_I,
        tau=2,
        rate_inflection=r0I,
        activation_exponent=0.9,
        bias=0,
        cap=12000.0,
    )

    # SST-like population: linear activation, slower dynamics
    pops["S"] = population_parameters(
        "S",
        size=size_S,
        tau=4,
        rate_inflection=r0S,
        activation_exponent=1.,
        bias=0,
        cap=6000.0,
    )

    return pops,{"E": r0E, "I": r0I, "S": r0S}


# ============================================================
# Compartment definitions
# ============================================================

def make_compartments(lp, pop, r0):
    """
    Build all compartment parameter dicts.

    Args:
        lp:    Learning parameter dict from make_learning_params().
        r0:  target average rates based on population settings (used to read r0 values for eta scaling).

    Returns:
        Dict of all compartment parameter dicts keyed by compartment id.
        Filtering to ACTIVE_COMPARTMENTS happens in build_net.
    """

    r0E = r0["E"] # target average for the population. 
    r0I = r0["I"]
    r0S = r0["S"]

    scale = lp["SCALE"]
    slow = lp["SLOW"]
    s = scale*slow
    t_calc = lp["TAU_CALC"]

    # Unpack frequently used learning params for readability
    TAU_BCM      = lp["TAU_BCM"]
    TAU_SLOW     = lp["TAU_SLOW"]
    TAUW         = lp["TAUW"]
    TAU_HOMEO_E  = lp["TAU_HOMEO_E"]; TAU_HOMEO_I = lp["TAU_HOMEO_I"]; TAU_HOMEO_S = lp["TAU_HOMEO_S"]
    TAUL_E       = lp["TAUL_E"];  TAUL_P  = lp["TAUL_P"];  TAUL_I  = lp["TAUL_I"]; TAUL_S = lp["TAUL_S"]
    DELTA_E      = lp["DELTA_E"]; DELTA_I = lp["DELTA_I"]; DELTA_S = lp["DELTA_S"]
    ZETA_P       = lp["ZETA_P"];  ZETA_E  = lp["ZETA_E"];  ZETA_I  = lp["ZETA_I"]; ZETA_S = lp["ZETA_S"]
    META_P       = lp["META_P"];  META_E  = lp["META_EE"];  META_I  = lp["META_EI"]; META_S = lp["META_II"]
    REG_P        = lp["REG_P"];   REG_E   = lp["REG_E"];   REG_I   = lp["REG_I"];  REG_S  = lp["REG_S"]
    BETA0        = lp["BETA0"]
    KAPPA_Q      = lp["KAPPA_Q"]
    Q_R          = lp["Q_R"]
    THETA_R      = lp["THETA_R"]
    RAD_E        = lp["RAD_E"]
    RAD_I        = lp["RAD_I"]
    RAD_S        = lp["RAD_S"]

    # P->E feedforward input fraction of total excitatory drive.
    # Used to normalize zeta so the amplitude learning rate is
    # comparable to other compartments regardless of input ratio.

    # This is one of the most important variables in the simulation, as this is the control knob for the feedforward strength in the network

    ap = 0.5
    an = 1.
    frac = 0.1
    d = DELTA_E*10
    d2 = -1.
    d3 = 1
    # I-I band power configuration for oscillatory amplitude learning.
    # Bands steer I-I amplitude based on relative power in fast/slow bands.
    I_I_band = {
        "amplitude": {
            "target": "I_I",
            "tau":  {"f": 2,  "m": 4, "s": 8},
            "taup": t_calc,
            "theta": {"f": [0.5, 0.55], "s": [0.25, 0.4]},
            "eta":   {"f": [d, 2*d],
                      "s": [2*d, d]},
        }
    }

    # S-I secondary band power configuration for oscillatory amplitude learning.
    # Bands steer I-I amplitude based on relative power in fast/slow bands.
    I_S_band = {
        "amplitude": {
            "target": "S_I",
            "tau":  {"f": 4,  "m": 10, "s": 25},
            "taup": TAUL_S,
            "theta": {"f": [0.5, 0.65], "s": [0.2, 0.5]},
            "eta":   {"f": [ZETA_S * d * d3, ZETA_S * d * d3],
                      "s": [ZETA_S * d * d3, ZETA_S * d * d3]},
        }
    }


    # This is one of the most important variables in the simulation, as this is the control knob for the feedforward strength in the network
    in_rec = lp["SCALE"]*0.5
    # A_PE is a feedfoward estimate based on MNIST averages
    A_PE = 1*r0E*in_rec*0.2
    # chosen amplitudes are renormalized such that the gain of the activation function at the target average is 1
    A_start = 30
    A_EE = A_start
    A_EI = A_start
    A_IE = -A_start
    A_II = -A_start
    A_ES = 10
    A_SE = -10
    A_IS = -5
    A_SI = -10

    comps = {}

    # ----------------------------------------------------------
    # P -> E: feedforward input from pixel layer to excitatory
    # Dense receptive field (3.5px radius, 50 synapses (both adjustable)) for full
    # local visibility. Amplitude targets ff/recurrent ratio via
    # ueff ratio learning. zeta divided by in_rec to normalize
    # learning rate relative to the 10% ff drive fraction.
    # ----------------------------------------------------------
    comps["P_E"] = compartment_parameters(
        id="P_E", source="P", target="E",
        ellipse=[3.5, 3.5], tsyn=50,
        A=A_PE, A0=A_PE,
        eta=lp["LR_PE"]/ r0E / 30,
        beta=lp["BETA_EE"], beta0=BETA0,
        bn=lp["BN_PE"], bp=lp["BP_PE"],
        an=lp["AN_PE"], ap=lp["AP_PE"],
        taul=t_calc, etal=META_P,
        etar=REG_P, kappa=KAPPA_Q, thetar=THETA_R,
        rho=0.,
        tau=1000, tauw=TAUW, taug=1000, taub=TAU_BCM,
        thetaz=A_PE * in_rec * 20.*0, z_value=in_rec,
        ratio="gain", c_c=["P_E","E_E"],
        zeta=DELTA_E*0.05,   # normalized by ff input fraction
        rin=1., rout=1., tauin=lp["TCI_PE"] , tauout=lp["TCO_PE"] ,
        delta=0.,           # no rate-target amp learning on ff input
        rate_target=r0E, eps=2.,
    )

    # ----------------------------------------------------------
    # E -> E: recurrent excitatory
    # Main driver of selectivity via Hebbian covariance learning.
    # Amplitude learning targets firing rate homeostasis (delta).
    # ----------------------------------------------------------
    comps["E_E"] = compartment_parameters(
        id="E_E", source="E", target="E",
        ellipse=[RAD_E, RAD_E], tsyn=50,
        A=A_EE, A0=A_EE,
        eta=lp["LR_EE"] / r0E / r0E,
        beta=lp["BETA_EE"], beta0=BETA0,
        bn=lp["BN_EE"], bp=lp["BP_EE"],
        an=lp["AN_EE"], ap=lp["AP_EE"],
        taul=t_calc, etal=lp["META_EE"],
        etar=REG_E, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_E"],
        tau=t_calc*1, tauw=TAUW, taug=t_calc*1, taub=TAU_HOMEO_E,
        #zeta=-DELTA_E*1, z_value=r0E*0.4*0.6, ratio="E2",c_c=["E_I","E_I"],
        rin=1., rout=0., tauin=lp["TCI_EE"] , tauout=lp["TCO_EE"] ,
        delta=DELTA_E*1., rate_target=r0E, eps=1.,
        stat=False,
        power={"tauf": TAU_BCM, "taus": TAU_SLOW},
        SST=SST(sst_type="E",target=["E_E","E_E"], omega=2, tau=[12,0.4,200,0.6]),
    )

    # ----------------------------------------------------------
    # E -> I: excitatory drive to inhibitory (PV-like)
    # Provides the excitatory input that drives PV firing.
    # Small amplitude plasticity toward inhibitory rate target.
    # ----------------------------------------------------------
    comps["E_I"] = compartment_parameters(
        id="E_I", source="E", target="I",
        ellipse=[RAD_I, RAD_I], tsyn=50,
        A=A_EI, A0=A_EI,
        eta=lp["LR_EI"] / r0E / r0I,
        beta=lp["BETA_EI"], beta0=BETA0,
        bn=lp["BN_EI"], bp=lp["BP_EI"],
        an=lp["AN_EI"], ap=lp["AP_EI"],
        taul=t_calc, etal=lp["META_EI"],
        etar=REG_E, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_E"]*0.2,
        tau=t_calc*0.1, tauw=TAUW, taug=t_calc*0.1, taub=TAU_BCM,
        zeta=DELTA_E*10, z_value=r0E, ratio="EPV",c_c=["E_I","E_I"],#what the z_value should be exactly to get the target right is unclear. STD reduces rates seen in the input, and the log-normal weighting skews the perceived mean.
        rin=1., rout=0., tauin=lp["TCI_EI"] , tauout=lp["TCO_EI"] ,
        delta=DELTA_E * 0, rate_target=r0I, eps=1.,
        SST=SST(sst_type="PV",target=["E_I","E_I"], omega=2, tau=[6,0.7,100,0.1]),
    )

    # ----------------------------------------------------------
    # E -> S: excitatory drive to SST population
    # SST post-synaptic gating via quantile-threshold nonlinearity.
    # ----------------------------------------------------------
    comps["E_S"] = compartment_parameters(
        id="E_S", source="E", target="S",
        ellipse=[RAD_S, RAD_S], tsyn=50,
        A=A_ES, A0=A_ES,
        eta=lp["LR_ES"] / r0S / r0S,
        beta=lp["BETA_ES"], beta0=BETA0,
        bn=lp["BN_ES"], bp=lp["BP_ES"],
        an=lp["AN_ES"], ap=lp["AP_ES"],
        taul=TAUL_E, etal=META_E,
        etar=REG_E, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_E"]*0.05,
        tau=TAU_HOMEO_E, tauw=TAUW, taug=TAUL_E, taub=TAU_BCM,
        zeta=0., z_value=0., ratio="E/I",
        rin=1., rout=1., tauin=lp["TCI_ES"], tauout=lp["TCO_ES"],
        delta=DELTA_E * 0.6, rate_target=r0S, eps=1.,
        SST=SST(sst_type="post",target=["E_S","E_S"], omega=2, tau=[20]),
    )

    # ----------------------------------------------------------
    # I -> E: inhibitory PV-like feedback to excitatory
    # Main inhibitory control of E population activity.
    # Amplitude learning drives E/I balance toward z_value target.
    # zeta/zeta2 asymmetry allows faster response to over-excitation/inhibition. This is in case the E/I ratio would lead to a consistent amplitude drift.
    # the zeta/zeta2 needs to be handled carefully to avoid the above stated drift.
    # this likely will also influence the width of the I_E amplitude distribution over the population.
    # the prefered asymmetry direction might not always be the same...
    # ----------------------------------------------------------
    comps["I_E"] = compartment_parameters(
        id="I_E", source="I", target="E",
        ellipse=[RAD_I, RAD_I], tsyn=20,
        A=A_IE, A0=-A_IE,
        eta=lp["LR_IE"] / r0I / r0E,
        beta=lp["BETA_IE"], beta0=BETA0,
        bn=lp["BN_IE"], bp=lp["BP_IE"],
        an=lp["AN_IE"], ap=lp["AP_IE"],
        taul=t_calc, etal=lp["META_IE"],
        etar=REG_I, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_I"],
        tau=t_calc, tauw=TAUW, taub=TAU_BCM,
        #zeta=DELTA_E, taug=TAU_HOMEO_E, z_value=1.5, thetaz=0.6, ratio="CV",c_c=["E_E","I_E"],
        zeta=DELTA_E * 5., taug=t_calc,
        z_value=0.2, ratio="E2", c_c=["E_E","I_E"],thetaz=4., # ZETA_E multiplier and z_value should be linked
        #z_value=0.2, ratio="sparse", c_c=["E_E","I_E"],
        rin=1., rout=0., tauin=lp["TCI_IE"] , tauout=lp["TCO_IE"] ,
        delta=0., rate_target=r0E, eps=1.,
        SST=SST(sst_type="PV",target=["E_E","I_E"], omega=2, tau=[6,0.7,100,0.4]),
    )

    # ----------------------------------------------------------
    # I -> I: recurrent inhibitory
    # Band-power amplitude learning steers oscillatory dynamics.
    # Correlation-based ratio learning balances I-I synchrony.
    # ----------------------------------------------------------
    comps["I_I"] = compartment_parameters(
        id="I_I", source="I", target="I",
        ellipse=[RAD_I, RAD_I], tsyn=20,
        A=A_II, A0=-A_II,
        eta=lp["LR_II"] / r0I / r0I,
        beta=lp["BETA_II"], beta0=BETA0,
        bn=lp["BN_II"], bp=lp["BP_II"],
        an=lp["AN_II"], ap=lp["AP_II"],
        taul=t_calc, etal=lp["META_II"],
        etar=REG_I, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_I"] * 0.01,
        tau=t_calc, tauw=TAUW, taub=TAU_SLOW,
        zeta=-DELTA_E * 1, taug=t_calc,
        z_value=0.5, ratio="gain", c_c=["E_I","I_I"],thetaz=0,
        #z_value=0.25, ratio="sparse", c_c=["E_I","I_I"],thetaz=0,
        #zeta=-DELTA_E*10, taug=t_calc, z_value=1., ratio="corr",c_c=["E_I","I_I"],
        rin=1., rout=1., tauin=lp["TCI_II"] , tauout=lp["TCO_II"] ,
        delta=0., rate_target=r0I, eps=1.,
        SST=SST(sst_type="PV",target=["I_I","I_I"], omega=2, tau=[6,0.7,100,0.1]),
        bands=I_I_band,
    )

    # ----------------------------------------------------------
    # I -> S: inhibitory input to SST population
    # ----------------------------------------------------------
    comps["I_S"] = compartment_parameters(
        id="I_S", source="I", target="S",
        ellipse=[RAD_I, RAD_I], tsyn=20,
        A=A_IS, A0=-A_IS,
        eta=lp["LR_IS"] / r0S / r0I / r0S,
        beta=lp["BETA_IS"], beta0=BETA0,
        bn=lp["BN_IS"], bp=lp["BP_IS"],
        an=lp["AN_IS"], ap=lp["AP_IS"],
        taul=TAUL_I, etal=META_I,
        etar=REG_I, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_I"],
        tau=TAU_HOMEO_I, tauw=TAUW, taub=TAU_SLOW, taug=TAUL_I,
        #zeta=-DELTA_I*10,z_value=0.4, ratio="supp", c_c=["E_S","I_S"],
        zeta=-DELTA_I*20,z_value=0.45, ratio="other", c_c=["E_S","I_S"],
        #zeta=DELTA_I * 5., z_value=0.15, ratio="corr",
        rin=1., rout=1., tauin=lp["TCI_IS"], tauout=lp["TCO_IS"],
        delta=-DELTA_I * 0., rate_target=r0S, eps=1.,
        bands=I_S_band,
    )

    # ----------------------------------------------------------
    # S -> E: SST-like dendritic inhibition of excitatory
    # Pre-synaptic SST gating: synapse learning driven by
    # dendritic burst (E_eff) of post-synaptic neuron.
    # Amplitude targets Ieff ratio to balance SST contribution.
    # ----------------------------------------------------------
    comps["S_E"] = compartment_parameters(
        id="S_E", source="S", target="E",
        ellipse=[RAD_S, RAD_S], tsyn=25,
        A=A_SE, A0=-A_SE,
        eta=lp["LR_SE"] / r0S / r0E,
        beta=lp["BETA_SE"], beta0=BETA0,
        bn=lp["BN_SE"], bp=lp["BP_SE"],
        an=lp["AN_SE"], ap=lp["AP_SE"],
        taul=TAUL_S, etal=META_S,
        etar=REG_S, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_I"]*0.1,
        tau=TAU_HOMEO_S, tauw=TAUW, taub=TAU_SLOW, taug=TAUL_S,
        zeta=-DELTA_S*30,z_value=0.5, ratio="supp", c_c=["E_E","S_E"],
        #zeta=-DELTA_S*30,z_value=0.55, ratio="other", c_c=["E_E","S_E"],
        #zeta=DELTA_S*50, z_value=0.9,thetaz=0.02, ratio="SST",
        rin=1., rout=1., tauin=lp["TCI_SE"], tauout=lp["TCO_SE"],
        delta=0., rate_target=r0E, eps=1.,
        SST=SST(sst_type="pre",target=["E_E","E_E"], omega=1.5),
    )

    # ----------------------------------------------------------
    # S -> I: SST disinhibition of inhibitory population
    # ----------------------------------------------------------
    comps["S_I"] = compartment_parameters(
        id="S_I", source="S", target="I",
        ellipse=[RAD_I, RAD_I], tsyn=25,
        A=A_SI, A0=-A_SI,
        eta=lp["LR_SI"] / r0S / r0I ,
        beta=lp["BETA_SI"], beta0=BETA0,
        bn=lp["BN_SI"], bp=lp["BP_SI"],
        an=lp["AN_SI"], ap=lp["AP_SI"],
        taul=TAUL_S, etal=META_S,
        etar=REG_S, kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_I"]*0.1,
        tau=TAU_HOMEO_S, tauw=TAUW, taub=TAU_SLOW, taug=TAUL_S,
        zeta=-DELTA_S*20,z_value=0.65, ratio="supp", c_c=["E_I","S_I"],
        #zeta=-DELTA_S*20,z_value=0.6, ratio="other", c_c=["E_I","S_I"],
        #zeta=DELTA_S*50, z_value=0.9,thetaz=0.02, ratio="SST",
        rin=1., rout=1., tauin=lp["TCI_SI"], tauout=lp["TCO_SI"],
        delta=0, rate_target=r0I, eps=1.,
        SST=SST(sst_type="pre",target=["E_I","I_I"], omega=1.5,tau=[2,6,16,128]),
    )

    # ----------------------------------------------------------
    # S -> S: SST recurrent inhibition
    # ----------------------------------------------------------
    comps["S_S"] = compartment_parameters(
        id="S_S", source="S", target="S",
        ellipse=[RAD_S, RAD_S], tsyn=25,
        A=-0.5, A0=0.5,
        eta=lp["LR_SS"] / r0S / r0S / r0S,
        beta=lp["BETA_SS"], beta0=BETA0,
        bn=lp["BN_SS"], bp=lp["BP_SS"],
        an=lp["AN_SS"], ap=lp["AP_SS"],
        kappa=KAPPA_Q, thetar=THETA_R, rq=Q_R,
        rho=lp["RHO_I"],
        tau=TAU_HOMEO_S, tauw=TAUW, taug=TAUL_S,
        zeta=DELTA_S, z_value=0.1, ratio="Ieff",
        rin=r0S * 1.5, rout=r0S * 0.5,
        tauin=-lp["TCI_II"], tauout=lp["TCO_II"],
        delta=0., rate_target=r0S, eps=0.,
    )

    return comps

def build_net(
    device=None,
    Z_E=16,
    frac_i=0.28,
    scale=0.1,
    active_compartments=None,
):
    """
    Build and return a Network instance.

    Args:
        device:               PyTorch device string or object.
        Z_E:                  Number of excitatory depth layers.
        frac_i:               Inhibitory depth fraction relative to E.
        scale:                Global learning rate scale factor.
        active_compartments:  List of compartment ids to include.
                              Defaults to ACTIVE_COMPARTMENTS.
                              Populations are inferred automatically
                              from active compartment source/target ids.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if active_compartments is None:
        active_compartments = ACTIVE_COMPARTMENTS

    size_E    = [28, 28, Z_E]
    all_pops,r0  = make_populations(size_E, frac_i)
    lp        = make_learning_params(
                    scale
                )
    all_comps = make_compartments(lp,all_pops,r0)

    # Filter to active topology
    comps = {k: v for k, v in all_comps.items() if k in active_compartments}

    # Infer required populations from active compartments
    required_pops = set()
    for c in comps.values():
        required_pops.add(c["source"])
        required_pops.add(c["target"])
    pops = {k: v for k, v in all_pops.items() if k in required_pops}

    return Network(device, pops, comps)