# test_network.py
import numpy as np
import torch
import matplotlib.pyplot as plt

from network import Network, population_parameters, compartment_parameters
from viz import (
    plot_volume, pop_volume,
    comp_band_phase, comp_band_amplitude,
    plot_compartment_weight_hist, plot_compartment_amplitude_hist, plot_compartment_amplitude_map
)

def main(device="cpu", steps=100000, seed=123):
    torch.manual_seed(seed)
    device = torch.device(device)

    # ----------------------- POPULATIONS -----------------------
    size = [24, 24, 1]

    pops = {
        "N": population_parameters(
            id="N", size=size, tau=2,
            rate_inflection=50, activation_exponent=1.0,
            baseline=0.0, cap=300.0
        ),
        "E": population_parameters(
            id="E", size=size, tau=5,             # E faster
            rate_inflection=58, activation_exponent=0.90,
            baseline=0.0, cap=300.0
        ),
        "I": population_parameters(
            id="I", size=size, tau=2,             # I slower
            rate_inflection=100, activation_exponent=1.3,
            baseline=0.0, cap=300.0
        ),
    }

    # ----------------------- BANDS -----------------------------
    bands = {"theta": {"period": 20, "tau": 30, "alpha": 0.9}}

    # Common homeostasis settings
    AVG_TAU = 250     # long avg for amplitude / targets (5–15 cycles)
    DELTA   = 5e-5    # slow amplitude learning
    RHO     = 4e-6    # pull to A0

    # ----------------------- COMPARTMENTS ----------------------
    comps = {}

    # N → E (feedforward drive; NO weight learning; standard target = 0)
    comps["N_E"] = compartment_parameters(
        id="N_E", source="N", target="E",
        ellipse=[3, 3], tsyn=16,
        A=2.0, A0=2.0,
        eta=0.0, beta=0.0,
        bands=bands, rho=RHO, tau=AVG_TAU,
        rin=0.0, rout=0.0, tauin=-1, tauout=-1,  # standard (no averaging; 0 target)
        delta=DELTA, rate_target=50.0
    )

    # E → E (Hebbian; standard centering = 0)
    comps["E_E"] = compartment_parameters(
        id="E_E", source="E", target="E",
        ellipse=[4, 4], tsyn=24,
        A=6.5, A0=6.5,
        eta=8e-5, beta=3e-5,
        bands=bands, rho=RHO, tau=AVG_TAU,
        rin=0.0, rout=0.0, tauin=-1, tauout=-1,  # standard (no averaging; 0 target)
        delta=DELTA, rate_target=75.0
    )

    # E → I (Hebbian; standard centering = 0)
    comps["E_I"] = compartment_parameters(
        id="E_I", source="E", target="I",
        ellipse=[4, 4], tsyn=24,
        A=4.0, A0=4.0,
        eta=7e-5, beta=3e-5,
        bands=bands, rho=RHO, tau=AVG_TAU,
        rin=0.0, rout=0.0, tauin=-1, tauout=-1,  # standard (no averaging; 0 target)
        delta=DELTA, rate_target=90.0
    )

    # I → E (post-error rule)
    comps["I_E"] = compartment_parameters(
        id="I_E", source="I", target="E",
        ellipse=[4, 4], tsyn=24,
        A=-5.5, A0=5.5,
        eta=8e-5, beta=3e-5,
        bands=bands, rho=RHO, tau=AVG_TAU,
        rin=0.0, rout=0.9, tauin=-1, tauout=300,   # post-error only
        delta=-DELTA, rate_target=65.0
    )

    # I → I (same logic)
    comps["I_I"] = compartment_parameters(
        id="I_I", source="I", target="I",
        ellipse=[4, 4], tsyn=24,
        A=-3.0, A0=3.0,
        eta=6e-5, beta=3e-5,
        bands=bands, rho=RHO, tau=AVG_TAU,
        rin=0.0, rout=1.1, tauin=-1, tauout=300,   # post-error only
        delta=-DELTA, rate_target=85.0
    )

    # ----------------------- BUILD NET -------------------------
    net = Network(device, pops, comps)

    # ----------------------- RUN -------------------------------
    # Persistent structured input with small temporal dither
    n = net.populations["N"].nneu
    base = 40.0 + 20.0 * torch.randn(n, device=device)   # frozen spatial bias
    base.clamp_(0, 120)

    for t in range(steps):
        drive = (base + 6.0 * torch.randn(n, device=device)).clamp(0, 300.0)
        net.populations["N"].rates = drive
        net.iterate()

        if (t % 2000) == 0 and t > 0:
            E = net.populations["E"].rates
            I = net.populations["I"].rates
            print(f"[t={t}] E mean={E.mean():.2f} std={E.std():.2f} | I mean={I.mean():.2f} std={I.std():.2f}")

    # ----------------------- PLOTS -----------------------------
    e_vol = pop_volume(net, "E", field="rates")
    i_vol = pop_volume(net, "I", field="rates")
    plot_volume(e_vol, mode="func", func=lambda a: np.max(a, axis=2), title="E rates (max)")
    plot_volume(i_vol, mode="func", func=lambda a: np.max(a, axis=2), title="I rates (max)")

    e_e = net.populations["E"].compartments["E_E"]
    phase_vol = comp_band_phase(e_e, "theta", side="out")
    amp_vol   = comp_band_amplitude(e_e, "theta", side="out", normalize_by_avg=True)
    plot_volume(phase_vol, mode="slice", z=0, title="E←E θ phase (z=0)", cmap="twilight")
    plot_volume(amp_vol,   mode="func", func=lambda a: a.mean(axis=2), title="E←E θ amplitude (mean z)")

    i_e = net.populations["E"].compartments["I_E"]
    e_i = net.populations["I"].compartments["E_I"]
    i_i = net.populations["I"].compartments["I_I"]

    plot_compartment_weight_hist(e_e, bins=80, title="Weights: E←E")
    plot_compartment_amplitude_hist(e_e, bins=60, title="Amplitudes: E←E")
    plot_compartment_amplitude_map(e_e, mode="func", func=lambda v: v.mean(axis=2), title="E←E amplitude (mean z)")
    #plot_compartment_amplitude_map(i_e, mode="func", func=lambda v: v.mean(axis=2), title="I←E amplitude (mean z)")
    #plot_compartment_amplitude_map(e_i, mode="func", func=lambda v: v.mean(axis=2), title="E←I amplitude (mean z)")
    #plot_compartment_amplitude_map(i_i, mode="func", func=lambda v: v.mean(axis=2), title="I←I amplitude (mean z)")

    plt.show()


if __name__ == "__main__":
    main()