#!/usr/bin/env python3
"""
viz_snapshot.py

Usage:
    python viz_snapshot.py --snapshot path/to/snapshot.pt --outdir figs_snapshot

Loads a saved Network snapshot and produces a set of diagnostic plots:
- Population firing rate distributions and spatial maps
- Compartment amplitude distributions and spatial maps
- Compartment weight distributions
- Compartment-level C, C2, CVt, CVs maps (for comp.stat=True)
- Band power fraction (Pf, Pm, Ps) distributions (where available)
"""

import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from network import Network  # assumes your classes are in network.py


# -------------------------
# Helpers
# -------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_cpu_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def reshape_to_grid(vec, size):
    """
    vec: 1D tensor/array of length N = W*H*Z
    size: [W, H, Z]
    returns: ndarray of shape (Z, H, W) for easy imshow with rows=H, cols=W
    """
    W, H, Z = size
    arr = to_cpu_np(vec).reshape(W, H, Z)
    # move Z to first axis so we can iterate slices as arr[z]
    return np.moveaxis(arr, -1, 0)  # (Z, H, W)


def plot_grid_slices(arr_zhw, title, outpath, vmin=None, vmax=None, cmap="viridis"):
    """
    arr_zhw: (Z, H, W)
    Draw each depth slice in a grid of subplots and save.
    """
    Z, H, W = arr_zhw.shape
    ncols = int(np.ceil(np.sqrt(Z)))
    nrows = int(np.ceil(Z / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for z in range(Z):
        r = z // ncols
        c = z % ncols
        ax = axes[r, c]
        im = ax.imshow(arr_zhw[z], origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"{title} (z={z})", fontsize=8)
        ax.axis("off")
    # hide any unused axes
    for z in range(Z, nrows * ncols):
        r = z // ncols
        c = z % ncols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def hist_plot(data, title, xlabel, outpath, bins=50, logy=False):
    arr = to_cpu_np(data).ravel()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(arr, bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# -------------------------
# Visualization functions
# -------------------------

def plot_population_rates(net, outdir):
    """Firing rate histograms + spatial maps per population."""
    for pid, pop in net.populations.items():
        r = pop.rates
        size = pop.size

        # Histogram
        hist_path = os.path.join(outdir, f"pop_{pid}_rates_hist.png")
        hist_plot(
            r,
            title=f"Population {pid} rates",
            xlabel="rate",
            outpath=hist_path,
            bins=60,
            logy=True,
        )

        # Spatial map(s)
        try:
            r_grid = reshape_to_grid(r, size)  # (Z,H,W)
            grid_path = os.path.join(outdir, f"pop_{pid}_rates_grid.png")
            plot_grid_slices(
                r_grid,
                title=f"Population {pid} rates (per depth slice)",
                outpath=grid_path,
                vmin=None,
                vmax=None,
                cmap="magma",
            )
        except Exception as e:
            print(f"[WARN] Could not reshape rates for pop {pid}: {e}")


def plot_compartment_amplitudes(net, outdir):
    """Amplitude histograms + spatial maps per compartment."""
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            a = comp.a
            size = pop.size

            base_name = f"comp_{pid}_{cid}_amp"

            # Hist
            hist_path = os.path.join(outdir, f"{base_name}_hist.png")
            hist_plot(
                a,
                title=f"Compartment {pid}:{cid} amplitudes",
                xlabel="amplitude",
                outpath=hist_path,
                bins=60,
                logy=True,
            )

            # Spatial
            try:
                a_grid = reshape_to_grid(a, size)
                grid_path = os.path.join(outdir, f"{base_name}_grid.png")
                plot_grid_slices(
                    a_grid,
                    title=f"Compartment {pid}:{cid} amplitudes",
                    outpath=grid_path,
                    cmap="viridis",
                )
            except Exception as e:
                print(f"[WARN] Could not reshape amplitudes for {pid}:{cid}: {e}")


def plot_compartment_weights(net, outdir):
    """Weight histograms per compartment."""
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            w = comp.w
            base_name = f"comp_{pid}_{cid}_w"

            hist_path = os.path.join(outdir, f"{base_name}_hist.png")
            hist_plot(
                w,
                title=f"Weights {pid}:{cid} (all synapses)",
                xlabel="w_ij",
                outpath=hist_path,
                bins=80,
                logy=True,
            )


def plot_compartment_stats(net, outdir):
    """C, C2, CVt, CVs maps per compartment (where comp.stat=True)."""
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            if not getattr(comp, "stat", False):
                continue

            size = pop.size

            try:
                C_grid = reshape_to_grid(comp.C, size)
                C2_grid = reshape_to_grid(comp.C2, size)
                CVt_grid = reshape_to_grid(comp.CVt, size)
                CVs_grid = reshape_to_grid(comp.CVs, size)
            except Exception as e:
                print(f"[WARN] Could not reshape stats for {pid}:{cid}: {e}")
                continue

            # C
            plot_grid_slices(
                C_grid,
                title=f"{pid}:{cid} C (local corr)",
                outpath=os.path.join(outdir, f"comp_{pid}_{cid}_C_grid.png"),
                cmap="coolwarm",
            )
            # |C| or C2-esque
            plot_grid_slices(
                C2_grid,
                title=f"{pid}:{cid} C2 (|C|-like)",
                outpath=os.path.join(outdir, f"comp_{pid}_{cid}_C2_grid.png"),
                cmap="plasma",
            )
            # CVt
            plot_grid_slices(
                CVt_grid,
                title=f"{pid}:{cid} CVt",
                outpath=os.path.join(outdir, f"comp_{pid}_{cid}_CVt_grid.png"),
                cmap="viridis",
            )
            # CVs
            plot_grid_slices(
                CVs_grid,
                title=f"{pid}:{cid} CVs",
                outpath=os.path.join(outdir, f"comp_{pid}_{cid}_CVs_grid.png"),
                cmap="viridis",
            )


def plot_band_power(net, outdir):
    """Plot band power fractions (Pf, Pm, Ps) where defined in comp.rate_band."""
    for pid, pop in net.populations.items():
        for cid, comp in pop.compartments.items():
            if not hasattr(comp, "rate_band"):
                continue

            # --- Amplitude band power ---
            if "amplitude" in comp.rate_band:
                amp = comp.rate_band["amplitude"]
                Pf = amp["p"]["f"].detach().cpu()
                Pm = amp["p"]["m"].detach().cpu()
                Ps = amp["p"]["s"].detach().cpu()
                Ptot = Pf + Pm + Ps + 1e-8

                for name, arr in [("f", Pf / Ptot), ("m", Pm / Ptot), ("s", Ps / Ptot)]:
                    hist_path = os.path.join(
                        outdir, f"comp_{pid}_{cid}_amp_band_{name}_frac_hist.png"
                    )
                    hist_plot(
                        arr,
                        title=f"{pid}:{cid} amplitude band {name} fraction",
                        xlabel=f"P_{name}/P_tot",
                        outpath=hist_path,
                        bins=50,
                        logy=False,
                    )

            # --- Synapse band power ---
            if "synapse" in comp.rate_band:
                # We’ll look at the OUT (postsynaptic) side for visualization
                out = comp.rate_band["synapse"]["out"]
                Pf = out["p"]["f"].detach().cpu()
                Pm = out["p"]["m"].detach().cpu()
                Ps = out["p"]["s"].detach().cpu()
                Ptot = Pf + Pm + Ps + 1e-8

                for name, arr in [("f", Pf / Ptot), ("m", Pm / Ptot), ("s", Ps / Ptot)]:
                    hist_path = os.path.join(
                        outdir, f"comp_{pid}_{cid}_syn_out_band_{name}_frac_hist.png"
                    )
                    hist_plot(
                        arr,
                        title=f"{pid}:{cid} synapse OUT band {name} fraction",
                        xlabel=f"P_{name}/P_tot",
                        outpath=hist_path,
                        bins=50,
                        logy=False,
                    )


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Network snapshot stats.")
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Path to saved network snapshot (.pt)")
    parser.add_argument("--outdir", type=str, default="figs_snapshot",
                        help="Directory to write figures to")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to map snapshot to (usually 'cpu' for viz)")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Load network
    device = torch.device(args.device)
    print(f"Loading snapshot from {args.snapshot} to device={device} ...")
    net = Network.load(args.snapshot, device=device)

    # Plot various statistics
    print("Plotting population firing rates...")
    plot_population_rates(net, args.outdir)

    print("Plotting compartment amplitudes...")
    plot_compartment_amplitudes(net, args.outdir)

    print("Plotting compartment weight histograms...")
    plot_compartment_weights(net, args.outdir)

    print("Plotting compartment correlation / CV maps...")
    plot_compartment_stats(net, args.outdir)

    print("Plotting band power fractions...")
    plot_band_power(net, args.outdir)

    print(f"Done. Figures saved in: {args.outdir}")


if __name__ == "__main__":
    main()