#!/usr/bin/env python3
"""
receptive_fields.py

Load a Network snapshot and visualize receptive fields for a given compartment.

Provides:
  - Random RFs (a handful of neurons)
  - RF column: all neurons at the same (x,y) across depth

Usage:
  python receptive_fields.py \
      --snapshot snapshots/net_step_50000.pt \
      --comp-id E_E \
      --rf-count 6 \
      --rf-column \
      --rf-x 14 --rf-y 14 \
      --outdir rf_figs_step_50000

Requires viz.py to define:
  - receptive_field_volume
  - plot_receptive_field
  - plot_receptive_field_column
  - index_to_xyz
"""

import os
import math
import argparse

import torch
import matplotlib.pyplot as plt

from viz import (
    plot_receptive_field,
    plot_receptive_field_column,
    index_to_xyz,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_network(snapshot_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    net = torch.load(snapshot_path, map_location=device)
    if hasattr(net, "device"):
        net.device = device
    return net


def find_compartment_by_id(net, comp_id):
    """
    Find first compartment with given id across all populations.
    """
    for pop in net.populations.values():
        for cid, comp in pop.compartments.items():
            if cid == comp_id:
                return comp
    return None


def plot_random_rfs(comp, rf_count, outdir, basename="rf_random"):
    """
    Plot rf_count random receptive fields from a given compartment.
    """
    n_targets = comp.target.nneu
    rf_count = min(rf_count, n_targets)
    if rf_count <= 0:
        print("No targets to plot RFs for; skipping random RF plots.")
        return

    rand_indices = torch.randperm(n_targets)[:rf_count].tolist()
    n = len(rand_indices)
    cols = min(n, 4)
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.reshape(rows, cols)

    for i, idx in enumerate(rand_indices):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        x_t, y_t, z_t = index_to_xyz(idx, comp.target.size)
        title = f"idx={idx}\n(x,y,z)=({x_t},{y_t},{z_t})"

        plot_receptive_field(
            comp,
            target_index=idx,
            mode="func",
            func=lambda v: v.max(axis=2),
            title=title,
            cmap="viridis",
            normalize=True,
            ax=ax,
        )

    # Turn off unused axes
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Random RFs for {comp.id} ({comp.sourceid}→{comp.targetid})", y=0.98)
    fig.tight_layout()
    fpath = os.path.join(outdir, f"{basename}_{comp.id}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"Saved random RFs to {fpath}")


def plot_column_rfs(comp, x, y, outdir, basename="rf_column"):
    """
    Plot RFs for all neurons at same (x,y) across depth in target geometry.
    """
    W_t, H_t, Z_t = comp.target.size
    x = max(0, min(W_t - 1, x))
    y = max(0, min(H_t - 1, y))

    fig, axes = plot_receptive_field_column(
        comp,
        x=x,
        y=y,
        mode="func",
        func=lambda v: v.max(axis=2),
        normalize=True,
        cmap="viridis",
    )
    fpath = os.path.join(outdir, f"{basename}_{comp.id}_x{x}_y{y}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"Saved RF column to {fpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize receptive fields from a saved Network snapshot."
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to .pt snapshot saved by net.save().",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading network.",
    )
    parser.add_argument(
        "--comp-id",
        type=str,
        default="E_E",
        help="Compartment id to visualize RFs for (e.g. 'E_E').",
    )
    parser.add_argument(
        "--rf-count",
        type=int,
        default=4,
        help="Number of random RFs to plot.",
    )
    parser.add_argument(
        "--rf-column",
        action="store_true",
        help="If set, also plot RFs for all depths at one (x,y) location.",
    )
    parser.add_argument(
        "--rf-x",
        type=int,
        default=None,
        help="x coordinate for RF column (if omitted, use center).",
    )
    parser.add_argument(
        "--rf-y",
        type=int,
        default=None,
        help="y coordinate for RF column (if omitted, use center).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Where to save RF figures. Default: <snapshot>_rf",
    )

    args = parser.parse_args()

    if args.outdir is None:
        base = os.path.splitext(os.path.basename(args.snapshot))[0]
        outdir = base + "_rf"
    else:
        outdir = args.outdir
    ensure_dir(outdir)

    print(f"Loading network from: {args.snapshot}")
    net = load_network(args.snapshot, device=args.device)

    comp = find_compartment_by_id(net, args.comp_id)
    if comp is None:
        print(f"Could not find compartment with id '{args.comp_id}'. Available:")
        for pid, pop in net.populations.items():
            for cid, c in pop.compartments.items():
                print(f"  - {cid} ({c.sourceid}->{c.targetid})")
        return

    print(f"Using compartment: {comp.id} ({comp.sourceid}->{comp.targetid})")

    # Random RFs
    plot_random_rfs(comp, args.rf_count, outdir)

    # Column RFs
    if args.rf_column:
        W_t, H_t, Z_t = comp.target.size
        x0 = W_t // 2 if args.rf_x is None else args.rf_x
        y0 = H_t // 2 if args.rf_y is None else args.rf_y
        plot_column_rfs(comp, x0, y0, outdir)


if __name__ == "__main__":
    main()