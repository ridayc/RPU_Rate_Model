"""
phase_trajectory_plot.py
=========================
Simplified trajectory plotter for the E/I MNIST network.

Unlike trajectory_inspector.py, this script focuses on exactly four plot
families, each shown per selected population:

  1. Per-neuron firing rates     — every (image, z) trace, colour = digit
  2. Mean rate per digit         — trace averaged over images and Z
  3. Mean rate per digit, per z  — trace averaged over images only, one
                                   line per (digit, z) combination.
                                   Colour encodes one axis, linestyle the
                                   other (see --color-by).
  4. Mean rate per z             — trace averaged over images and digits,
                                   one line per z value (complement of #2).

Each trial runs through three phases on a single step axis:

  warm-up   : no stimulus (P population held at 0)
  image     : stimulus presented (P population = image)
  shutdown  : stimulus removed again (P population held at 0)

Phase boundaries are drawn as shaded bands on every subplot so you can see
how rates rise after stimulus onset and decay after offset.

Population rate layout
-----------------------
Population rate buffers are stored flat in (W, H, Z) order, i.e. the flat
offset for spatial location (cx, cy) and feature z is:

    offset = (cx * H + cy) * Z + z

This module indexes into that buffer accordingly when probing locations.

State reset before each image
-------------------------------
Every image is an independent trial: the ENTIRE network's state -- every
population's rates, plus every compartment's lrates and SST.gavg running
average where present -- is always reset before that image's warm-up
phase begins. This covers the whole network, not just the populations
being recorded/plotted, since any state left un-reset (e.g. a running
average) can otherwise leak from one image's trial into the next and make
deterministic trials look digit-dependent before the image is ever shown.
Warm-up itself is unrelated to the reset -- it's just stimulus-off steps
that run after the reset, before the image is presented. The --reset-mode
flag controls what the reset target is:

  --reset-mode loaded   (default) Reset every image to the full network
                         state at the time this script loaded the network
                         (e.g. straight out of a snapshot), captured once
                         up front and reused for every image.

  --reset-mode fresh     Reset every image to all-zero state (rates,
                         lrates, and gavg all zeroed).

Usage
-----
    python phase_trajectory_plot.py --load-snapshot mnist_net_final.pt \\
        --mnist-root ./data \\
        --probe-xy 14,14 \\
        --n-warmup-steps 10 \\
        --n-image-steps 30 \\
        --n-shutdown-steps 10 \\
        --n-images 20 \\
        --rate-transform log1p \\
        --output-dir ./phase_out

Multiple locations:
    --probe-xy 14,14;4,4;23,23

Choosing populations (default: E, plus I if present):
    --populations E,I,M

Digit filter:
    --digits 3,8

Reset behaviour:
    --reset-mode fresh                 (start every image from all-zero state)
    --reset-mode loaded                (default: start every image from the
                                         state the network had at load time)

All session construction flags (--device, --Z-E, --frac-i, --scale, etc.)
are also accepted, same as trajectory_inspector.py.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Optional

from network_session import build_session, session_arg_parser


# ============================================================
# MNIST loader (unchanged behaviour from trajectory_inspector.py)
# ============================================================

def load_mnist_sample(mnist_root: str, n_images: int,
                       digits: Optional[List[int]] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (images, labels):
        images : float32 (n, 784)  values in [0, 255]
        labels : int64   (n,)
    Samples evenly across requested digit classes.
    """
    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms
        ds = MNIST(root=mnist_root, train=False, download=False,
                   transform=transforms.ToTensor())
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST from '{mnist_root}': {e}")

    classes = digits if digits is not None else list(range(10))
    per_class = max(1, n_images // len(classes))

    buckets: Dict[int, List] = {c: [] for c in classes}
    for img, label in ds:
        if label not in buckets:
            continue
        if len(buckets[label]) < per_class:
            buckets[label].append((img.numpy().flatten() * 255.0, label))
        if all(len(v) >= per_class for v in buckets.values()):
            break

    all_items = [item for v in buckets.values() for item in v]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_items))
    all_items = [all_items[i] for i in idx]

    images = np.stack([x[0] for x in all_items], axis=0).astype(np.float32)
    labels = np.array([x[1] for x in all_items], dtype=np.int64)
    return images, labels


# ============================================================
# Core: record one image through warm-up / image / shutdown
# ============================================================

def record_phase_trajectory(session, img_flat: np.ndarray,
                             probe_xy: List[Tuple[int, int]],
                             n_warmup: int, n_image: int, n_shutdown: int, cycles: int, normalize: bool,
                             pop_ids: List[str],
                             reset_state: Dict[str, object]
                             ) -> Dict[str, np.ndarray]:
    """
    Run one trial: reset the ENTIRE network (every population's rates,
    plus every compartment's lrates and SST.gavg where present) to
    reset_state, then run n_warmup steps with no stimulus, n_image steps
    with the stimulus presented, and n_shutdown steps with the stimulus
    removed again.

    pop_ids selects which populations (besides the input population "P")
    to record and return -- this only affects what gets returned/plotted,
    not what gets reset; the reset always covers the whole network so no
    hidden state (e.g. running averages or learning-rate traces) can leak
    from one image's trial into the next.

    reset_state is a snapshot produced by _clone_state or _zero_state.
    The reset happens before the warm-up phase begins -- warm-up itself
    never determines the reset state, it's just stimulus-off steps that
    run after the reset, before the image is presented.

    Returns dict pop_id -> float32 (n_locs, Z, T) where
    T = n_warmup + n_image + n_shutdown.
    """
    net = session.net
    Z = {p: net.populations[p].size[2] for p in pop_ids}
    sizes = {p: net.populations[p].size for p in pop_ids}  # (W, H, Z)
    n_locs = len(probe_xy)
    T = (n_warmup + n_image + n_shutdown)*cycles

    _restore_state(net, reset_state)

    P_rates = net.populations["P"].rates
    dev   = P_rates.device if isinstance(P_rates, torch.Tensor) else None
    dtype = P_rates.dtype  if isinstance(P_rates, torch.Tensor) else torch.float32
    stim = torch.tensor(img_flat, dtype=dtype, device=dev)
    zero_stim = torch.zeros_like(stim)

    trajs = {p: np.zeros((n_locs, Z[p], T), dtype=np.float32)
             for p in pop_ids}

    def set_input(on: bool):
        net.populations["P"].rates[:] = stim if on else zero_stim

    def record_step(step: int):
        for p in pop_ids:
            rates = net.populations[p].rates
            if isinstance(rates, torch.Tensor):
                if(not normalize):
                    rates = rates.cpu().numpy()
                else:
                    rates = rates.cpu().numpy()/(net.populations[p].compartments["E_"+p].rate_average+1e-9).cpu().numpy()
            W, H, Zp = sizes[p]
            for li, (cx, cy) in enumerate(probe_xy):
                # Rate buffers are stored flat in (W, H, Z) order, so the
                # flat offset for spatial location (cx, cy) is
                # (cx * H + cy) * Zp, NOT (cy * W + cx) * Zp.
                base = (cx * H + cy) * Zp
                trajs[p][li, :, step] = rates[base: base + Zp]

    step = 0
    for _ in range(0,cycles):
        set_input(False)
        for _ in range(n_warmup):
            session.step()
            record_step(step)
            step += 1

        set_input(True)
        for _ in range(n_image):
            session.step()
            record_step(step)
            step += 1

        set_input(False)
        for _ in range(n_shutdown):
            session.step()
            record_step(step)
            step += 1

    return trajs


def _clone_state(net) -> Dict[str, object]:
    """
    Snapshot ALL stateful variables on every population in the network --
    not just .rates, but also each compartment's lrates and (where
    present) its SST.gavg running average. Anything left out here is
    exactly the kind of hidden state that can silently leak between
    trials, so this snapshots everything stateful, regardless of which
    populations are actually being recorded/plotted.
    """
    state = {}
    for p in net.populations.values():
        state[p.id] = {}
        r = p.rates
        state[p.id]["rates"] = r.clone() if isinstance(r, torch.Tensor) else r.copy()
        for c in p.compartments.values():
            state[p.id][c.id] = {}
            lr = c.lrates
            state[p.id][c.id]["lrates"] = (lr.clone() if isinstance(lr, torch.Tensor)
                                            else lr.copy())
            if c.SST is not None and c.SST.type != "pre":
                g = c.SST.gavg
                state[p.id][c.id]["gavg"] = (g.clone() if isinstance(g, torch.Tensor)
                                              else g.copy())
    return state


def _restore_state(net, state: Dict[str, object]):
    """Write a snapshot produced by _clone_state back into the network."""
    for p in net.populations.values():
        p.rates[:] = state[p.id]["rates"]
        for c in p.compartments.values():
            c.lrates[:] = state[p.id][c.id]["lrates"]
            if c.SST is not None and c.SST.type != "pre":
                c.SST.gavg[:] = state[p.id][c.id]["gavg"]


def _zero_state(net) -> Dict[str, object]:
    """
    Build an all-zero snapshot with the same structure as _clone_state,
    for "fresh" reset mode -- so rates, lrates, AND gavg all start from
    zero, rather than only zeroing rates and leaving other state at
    whatever the network last had.
    """
    state = {}
    for p in net.populations.values():
        state[p.id] = {}
        r = p.rates
        state[p.id]["rates"] = (torch.zeros_like(r) if isinstance(r, torch.Tensor)
                                 else np.zeros_like(r))
        for c in p.compartments.values():
            state[p.id][c.id] = {}
            lr = c.lrates
            state[p.id][c.id]["lrates"] = (torch.zeros_like(lr) if isinstance(lr, torch.Tensor)
                                            else np.zeros_like(lr))
            if c.SST is not None and c.SST.type != "pre":
                g = c.SST.gavg
                state[p.id][c.id]["gavg"] = (torch.zeros_like(g) if isinstance(g, torch.Tensor)
                                              else np.zeros_like(g))
    return state


def collect_phase_trajectories(session, images: np.ndarray, labels: np.ndarray,
                                probe_xy: List[Tuple[int, int]],
                                n_warmup: int, n_image: int, n_shutdown: int, cycles: int, normalize: bool,
                                pop_ids: List[str],
                                reset_mode: str = "loaded"
                                ) -> Tuple[Dict, np.ndarray]:
    """
    Run record_phase_trajectory for every image. Every image is an
    independent trial: the ENTIRE network's state (all populations'
    rates, plus every compartment's lrates and SST.gavg where present) is
    reset to a baseline before that image's warm-up phase begins (warm-up
    is just stimulus-off steps that let the reset state evolve before the
    image is presented; it never determines the reset state itself).
    Resetting the whole network -- not just the recorded populations --
    matters because hidden state like running-average compartments can
    otherwise carry over from one image's trial into the next, making
    "deterministic" trials look digit-dependent even before the image is
    ever shown.

    reset_mode selects what that baseline is:
        "loaded" (default) : the full network state at the time this
            function was called -- i.e. however the network came out of
            loading (snapshot, prior session state, etc.), captured once
            up front and reused as the reset target for every image.
        "fresh"             : all state (rates, lrates, gavg) zeroed, for
            every image.

    Returns:
        all_trajs : dict  pop_id -> (n_imgs, n_locs, Z, T)
        labels    : (n_imgs,)
    """
    if reset_mode not in ("loaded", "fresh"):
        raise ValueError(f"Unknown reset_mode '{reset_mode}', "
                          f"expected 'loaded' or 'fresh'.")

    net = session.net
    Z = {p: net.populations[p].size[2] for p in pop_ids}
    n_locs = len(probe_xy)
    n_imgs = len(images)
    T = (n_warmup + n_image + n_shutdown)*cycles

    all_trajs = {p: np.zeros((n_imgs, n_locs, Z[p], T), dtype=np.float32)
                 for p in pop_ids}

    saved_freeze = net.freeze
    net.freeze = True

    # Full original state of the WHOLE network, as it came out of loading.
    # Restored at the very end no matter what, regardless of reset_mode.
    original_state = _clone_state(net)

    if reset_mode == "loaded":
        baseline_state = original_state
    else:  # "fresh"
        baseline_state = _zero_state(net)

    try:
        for ii, (img, lbl) in enumerate(zip(images, labels)):
            print(f"  image {ii + 1:3d}/{n_imgs}  (digit {lbl})")
            traj = record_phase_trajectory(session, img, probe_xy,
                                            n_warmup, n_image, n_shutdown, cycles, normalize,
                                            pop_ids, reset_state=baseline_state)
            for p in pop_ids:
                all_trajs[p][ii] = traj[p]
    finally:
        _restore_state(net, original_state)
        net.freeze = saved_freeze

    return all_trajs, labels


# ============================================================
# Rate transform (applied at plot time only)
# ============================================================

def apply_transform(r: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return r
    if mode == "log":
        return np.log(np.clip(r, 1e-2, None))
    if mode == "log1p":
        return np.log1p(np.clip(r, 0, None))
    if mode == "pow":
        return np.power(np.clip(r, 0, None),0.25)
    raise ValueError(f"Unknown rate transform '{mode}'")


TRANSFORM_YLABEL = {
    "none":  "rate",
    "log":   "log(rate)",
    "log1p": "log(1 + rate)",
    "pow":  "pow(rate)",
}


# ============================================================
# Plotting
# ============================================================

C_BG  = "#0b0d14"
C_AX  = "#141720"
C_TXT = "#c8cce0"
C_WARMUP   = "#1c2030"
C_SHUTDOWN = "#1c2030"
DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _style(a, title=""):
    a.set_facecolor(C_AX)
    a.tick_params(colors=C_TXT, labelsize=8)
    for sp in a.spines.values():
        sp.set_edgecolor("#2a2e44")
    a.set_title(title, color=C_TXT, fontsize=9, pad=4)
    a.xaxis.label.set_color(C_TXT)
    a.yaxis.label.set_color(C_TXT)


def _mark_phases(ax, n_warmup: int, n_image: int, n_shutdown: int,
                  n_cycles: int = 1):
    """
    Shade warm-up / shutdown bands and draw boundary lines, repeated once
    per cycle so every warmup->image->shutdown repetition in the trace is
    marked, not just the first.
    """
    cycle_len = n_warmup + n_image + n_shutdown
    for c in range(n_cycles):
        c0    = c * cycle_len
        t_on  = c0 + n_warmup
        t_off = c0 + n_warmup + n_image
        t_end = c0 + cycle_len
        ax.axvspan(c0, t_on,    color=C_WARMUP,   alpha=0.5, zorder=0)
        ax.axvspan(t_off, t_end, color=C_SHUTDOWN, alpha=0.5, zorder=0)
        ax.axvline(t_on,  color="#7eb8f7", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(t_off, color="#f7a07e", lw=1.0, ls="--", alpha=0.8)
        # Cycle boundary (end of shutdown / start of next warm-up), skip
        # drawing it after the very last cycle since there's no "next".
        if c < n_cycles - 1:
            ax.axvline(t_end, color="#5c6178", lw=1.0, ls=":", alpha=0.6)


def plot_per_neuron(ax, traj_loc_pop: np.ndarray, labels: np.ndarray,
                     pop_id: str, loc_label: str, transform: str,
                     n_warmup: int, n_image: int, n_shutdown: int,
                     n_cycles: int = 1):
    """
    Every (image, z) trace overlaid, colour = digit.
    traj_loc_pop shape: (n_imgs, Z, T)
    """
    n_imgs, Z, T = traj_loc_pop.shape
    steps = np.arange(T)
    data = apply_transform(traj_loc_pop, transform)
    for ii in range(n_imgs):
        col = DIGIT_COLORS[labels[ii] % 10]
        for z in range(Z):
            ax.plot(steps, data[ii, z], color=col, alpha=0.25, lw=0.8)

    from matplotlib.patches import Patch
    present = sorted(set(labels.tolist()))
    handles = [Patch(color=DIGIT_COLORS[d], label=str(d)) for d in present]
    ax.legend(handles=handles, fontsize=6, labelcolor=C_TXT,
              facecolor=C_AX, edgecolor="none",
              ncol=min(5, len(present)), loc="upper right")

    _mark_phases(ax, n_warmup, n_image, n_shutdown, n_cycles)
    ax.set_xlabel("step")
    ax.set_ylabel(TRANSFORM_YLABEL[transform])
    _style(ax, f"Per-neuron rates [{pop_id}] @ {loc_label}")


def plot_mean_per_digit(ax, traj_loc_pop: np.ndarray, labels: np.ndarray,
                         pop_id: str, loc_label: str, transform: str,
                         n_warmup: int, n_image: int, n_shutdown: int,
                         n_cycles: int = 1):
    """
    Mean trace per digit, averaged over images and Z neurons.
    traj_loc_pop shape: (n_imgs, Z, T)
    """
    T = traj_loc_pop.shape[2]
    steps = np.arange(T)
    present = sorted(set(labels.tolist()))
    for d in present:
        mask = labels == d
        mean_traj = traj_loc_pop[mask].mean(axis=(0, 1))   # (T,)
        mean_traj = apply_transform(mean_traj, transform)
        ax.plot(steps, mean_traj, color=DIGIT_COLORS[d], lw=1.8, label=str(d))
    ax.legend(fontsize=6, labelcolor=C_TXT, facecolor=C_AX, edgecolor="none",
              ncol=5, loc="upper right")

    _mark_phases(ax, n_warmup, n_image, n_shutdown, n_cycles)
    ax.set_xlabel("step")
    ax.set_ylabel(TRANSFORM_YLABEL[transform])
    _style(ax, f"Mean rate per digit [{pop_id}] @ {loc_label}")


def plot_mean_per_z(ax, traj_loc_pop: np.ndarray, labels: np.ndarray,
                     pop_id: str, loc_label: str, transform: str,
                     n_warmup: int, n_image: int, n_shutdown: int,
                     n_cycles: int = 1):
    """
    Mean trace per z, averaged over images and digits (i.e. pooled across
    everything except Z) — the complement of plot_mean_per_digit, which
    pools Z and keeps digit separate. One line per z value, plus a dashed
    white "all z" reference line.
    traj_loc_pop shape: (n_imgs, Z, T)
    """
    n_imgs, Z, T = traj_loc_pop.shape
    steps = np.arange(T)
    z_colors = plt.cm.viridis(np.linspace(0, 0.9, max(Z, 1)))

    for z in range(Z):
        mean_traj = apply_transform(traj_loc_pop[:, z, :].mean(axis=0), transform)
        ax.plot(steps, mean_traj, color=z_colors[z], lw=1.8, label=f"z={z}")

    overall = apply_transform(traj_loc_pop.mean(axis=(0, 1)), transform)
    ax.plot(steps, overall, color="white", lw=1.2, ls="--", alpha=0.6,
            label="all z")

    ax.legend(fontsize=6, labelcolor=C_TXT, facecolor=C_AX, edgecolor="none",
              ncol=min(6, Z + 1), loc="upper right")

    _mark_phases(ax, n_warmup, n_image, n_shutdown, n_cycles)
    ax.set_xlabel("step")
    ax.set_ylabel(TRANSFORM_YLABEL[transform])
    _style(ax, f"Mean rate per z [{pop_id}] @ {loc_label}")


LINESTYLES = ["-", "--", ":", "-.",
              (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2)), (0, (4, 1))]


def _linestyle_for(idx: int) -> str:
    return LINESTYLES[idx % len(LINESTYLES)]


def plot_mean_per_digit_per_z(ax, traj_loc_pop: np.ndarray, labels: np.ndarray,
                               pop_id: str, loc_label: str, transform: str,
                               n_warmup: int, n_image: int, n_shutdown: int,
                               color_by: str = "digit", n_cycles: int = 1):
    """
    Mean trace per (digit, z) — averaged over images only, kept separate
    per Z. One line per combination.

    color_by = "digit": colour encodes digit, linestyle encodes z.
               Better when there are few Z values but many digits.
    color_by = "z":     colour encodes z, linestyle encodes digit.
               Better when there are many Z values but few digits.

    traj_loc_pop shape: (n_imgs, Z, T)
    """
    n_imgs, Z, T = traj_loc_pop.shape
    steps = np.arange(T)
    present = sorted(set(labels.tolist()))
    z_colors = plt.cm.viridis(np.linspace(0, 0.9, max(Z, 1)))

    for d in present:
        mask = labels == d
        # mean over images of this digit only, kept per z: (Z, T)
        mean_per_z = traj_loc_pop[mask].mean(axis=0)
        mean_per_z = apply_transform(mean_per_z, transform)
        for z in range(Z):
            if color_by == "digit":
                color = DIGIT_COLORS[d % 10]
                style = _linestyle_for(z)
            else:
                color = z_colors[z]
                style = _linestyle_for(present.index(d))
            ax.plot(steps, mean_per_z[z], color=color, ls=style, lw=1.5,
                    alpha=0.9)

    # Two-part legend: colour key + linestyle key, so it stays readable
    # instead of listing every (digit, z) combination.
    from matplotlib.lines import Line2D
    if color_by == "digit":
        color_handles = [Line2D([0], [0], color=DIGIT_COLORS[d % 10], lw=2,
                                 label=f"digit {d}") for d in present]
        style_handles = [Line2D([0], [0], color=C_TXT, lw=1.5,
                                 ls=_linestyle_for(z), label=f"z={z}")
                         for z in range(Z)]
    else:
        color_handles = [Line2D([0], [0], color=z_colors[z], lw=2,
                                 label=f"z={z}") for z in range(Z)]
        style_handles = [Line2D([0], [0], color=C_TXT, lw=1.5,
                                 ls=_linestyle_for(i), label=f"digit {d}")
                         for i, d in enumerate(present)]

    leg1 = ax.legend(handles=color_handles, fontsize=6, labelcolor=C_TXT,
                     facecolor=C_AX, edgecolor="none", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=6, labelcolor=C_TXT,
              facecolor=C_AX, edgecolor="none", loc="lower left",
              bbox_to_anchor=(1.01, 0.0), borderaxespad=0)

    _mark_phases(ax, n_warmup, n_image, n_shutdown, n_cycles)
    ax.set_xlabel("step")
    ax.set_ylabel(TRANSFORM_YLABEL[transform])
    _style(ax, f"Mean rate per digit, per z [{pop_id}] @ {loc_label}")


def make_figure(all_trajs: Dict, labels: np.ndarray,
                 probe_xy: List[Tuple[int, int]],
                 n_warmup: int, n_image: int, n_shutdown: int,
                 save_path: str, transform: str = "none",
                 color_by: str = "digit", n_cycles: int = 1):

    pop_ids = list(all_trajs.keys())
    n_rows = 4  # per-neuron / mean-per-digit / mean-per-digit-per-z / mean-per-z
    n_cols = len(pop_ids)  # one column per population (E, [I], ...)

    for li, (cx, cy) in enumerate(probe_xy):
        loc_label = f"({cx},{cy})"
        print(f"[plot] Generating figure for location {loc_label} ...")

        fig = plt.figure(figsize=(9 * n_cols, 4.5 * n_rows), facecolor=C_BG)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                hspace=0.5, wspace=0.55)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                for r in range(n_rows)]

        for ci, p in enumerate(pop_ids):
            t = all_trajs[p][:, li, :, :]   # (n_imgs, Z, T)
            plot_per_neuron(axes[0][ci], t, labels, p, loc_label, transform,
                             n_warmup, n_image, n_shutdown, n_cycles)
            plot_mean_per_digit(axes[1][ci], t, labels, p, loc_label, transform,
                                 n_warmup, n_image, n_shutdown, n_cycles)
            plot_mean_per_digit_per_z(axes[2][ci], t, labels, p, loc_label,
                                       transform, n_warmup, n_image, n_shutdown,
                                       color_by, n_cycles)
            plot_mean_per_z(axes[3][ci], t, labels, p, loc_label, transform,
                             n_warmup, n_image, n_shutdown, n_cycles)

        fig.patch.set_facecolor(C_BG)
        cycle_note = f", cycles={n_cycles}" if n_cycles != 1 else ""
        fig.suptitle(
            f"Phase Trajectory Plot — location {loc_label}  "
            f"(warm-up={n_warmup}, image={n_image}, shutdown={n_shutdown}"
            f"{cycle_note})",
            color=C_TXT, fontsize=12, y=0.999)

        fname = os.path.join(save_path, f"phase_traj_{cx}_{cy}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)
        print(f"[plot] Saved -> {fname}")


# ============================================================
# Main
# ============================================================

def resolve_population_ids(net, requested: Optional[List[str]]) -> List[str]:
    """
    Decide which populations (other than the input population "P") to
    record and plot.

    - If `requested` is given, use exactly those IDs, in that order, after
      checking each one actually exists on the network.
    - Otherwise, fall back to the old default: "E", plus "I" if present.
      (Any other populations the network might have — e.g. a modulatory
      "M" population — are NOT auto-included; pass --populations to get
      them.)
    """
    available = [p for p in net.populations.keys() if p != "P"]

    if requested is None:
        return ["E"] + (["I"] if "I" in net.populations else [])

    missing = [p for p in requested if p not in net.populations]
    if missing:
        raise ValueError(
            f"Requested population(s) {missing} not found on the network. "
            f"Available populations: {available}"
        )
    return list(requested)


def plot_phase_trajectories(
    session,
    mnist_root: str       = "./data",
    n_images: int         = 20,
    digits: Optional[List[int]] = None,
    n_warmup_steps: int   = 10,
    n_image_steps: int    = 30,
    n_shutdown_steps: int = 10,
    n_cycles: int         = 1,  
    probe_xy              = None,
    rate_transform: str   = "none",
    populations: Optional[List[str]] = None,
    color_by: str         = "digit",
    normalize: bool         = False,
    reset_mode: str       = "loaded",
    output_dir: str       = ".",
) -> Dict:
    net = session.net
    W, H, Z_E = net.populations["E"].size

    if probe_xy is None:
        probe_xy = [(W // 2, H // 2)]

    pop_ids = resolve_population_ids(net, populations)

    print("\n=== Phase Trajectory Plot ===")
    print(f"  Locations        : {probe_xy}")
    print(f"  Warm-up steps    : {n_warmup_steps}")
    print(f"  Image steps      : {n_image_steps}")
    print(f"  Shutdown steps   : {n_shutdown_steps}")
    print(f"  Number of cycles : {n_cycles}")
    print(f"  Images           : {n_images}")
    print(f"  Digits filter    : {digits if digits else 'all'}")
    print(f"  Rate transform   : {rate_transform}")
    print(f"  Populations      : {pop_ids}")
    print(f"  Color encodes    : {color_by}  (linestyle encodes the other)")
    print(f"  Reset mode       : {reset_mode}  (reset happens before every "
          f"image's warm-up)")

    print("\nLoading MNIST sample ...")
    images, labels = load_mnist_sample(mnist_root, n_images, digits)
    print(f"  Loaded {len(images)} images, "
          f"classes present: {sorted(set(labels.tolist()))}")

    print("\nRecording phase trajectories ...")
    all_trajs, labels = collect_phase_trajectories(
        session, images, labels, probe_xy,
        n_warmup_steps, n_image_steps, n_shutdown_steps,n_cycles, normalize, pop_ids,
        reset_mode=reset_mode)

    os.makedirs(output_dir, exist_ok=True)
    make_figure(all_trajs, labels, probe_xy,
                n_warmup_steps, n_image_steps, n_shutdown_steps,
                output_dir, rate_transform, color_by, n_cycles)

    return dict(trajs=all_trajs, labels=labels, probe_xy=probe_xy)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simplified warm-up/image/shutdown phase trajectory "
                    "plotter for recurrent E/I MNIST network.",
        parents=[session_arg_parser()],
    )

    parser.add_argument("--mnist-root", type=str, default="./data",
                        help="Root directory for MNIST data.")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Total images to present (spread across classes).")
    parser.add_argument("--digits", type=str, default=None,
                        help="Comma-separated digit classes to include, e.g. "
                             "'3,8'.  Default: all 10 classes.")
    parser.add_argument("--n-warmup-steps", type=int, default=10,
                        help="Steps with no stimulus before the image phase.")
    parser.add_argument("--n-image-steps", type=int, default=30,
                        help="Steps with the stimulus presented.")
    parser.add_argument("--n-shutdown-steps", type=int, default=10,
                        help="Steps with no stimulus after the image phase.")
    parser.add_argument("--n-cycles", type=int, default=1,
                        help="Number of times to loop full rounds of warmup, image presentation and cooldown")
    parser.add_argument("--probe-xy", type=str, default=None,
                        help="Locations as 'x1,y1;x2,y2;...'.  "
                             "Default: image centre.")
    parser.add_argument("--populations", type=str, default=None,
                        help="Comma-separated population IDs to plot, e.g. "
                             "'E,I' or 'E,I,M'.  Must match population "
                             "names on the network (besides 'P', the input "
                             "population, which is never plotted).  "
                             "Default: E, plus I if present.")
    parser.add_argument("--rate-transform", type=str, default="none",
                        choices=["none", "log", "log1p", "pow"],
                        help="Transform applied to rates before plotting "
                             "(computed at plot time, not during recording).")
    parser.add_argument("--color-by", type=str, default="digit",
                        choices=["digit", "z"],
                        help="In the 'mean per digit, per z' panel: which "
                             "axis gets distinct colors (the other gets "
                             "distinct linestyles). 'digit' (default) is "
                             "usually clearer when Z is small; 'z' is "
                             "usually clearer when Z is large and few "
                             "digits are selected.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the rates by dividing by each neurons long term averages"
    )
    parser.add_argument(
        "--reset-mode", type=str, default="loaded",
        choices=["loaded", "fresh"],
        help="What full network state (rates, lrates, and gavg on every "
             "population, not just the ones being plotted) to reset to "
             "before every image's warm-up phase (reset always happens; "
             "this only controls the target). 'loaded' (default): the "
             "state the network was in when this script loaded it, "
             "reused for every image. 'fresh': everything zeroed for "
             "every image."
    )
    parser.add_argument("--output-dir", type=str, default="./phase_out",
                        help="Directory for output figures.")

    args = parser.parse_args()
    args.hdf5_n_snapshots = 1
    args.rate_n_snapshots = 1

    digits = ([int(d) for d in args.digits.split(",")]
              if args.digits else None)

    probe_xy = None
    if args.probe_xy:
        probe_xy = []
        for pair in args.probe_xy.split(";"):
            x, y = pair.strip().split(",")
            probe_xy.append((int(x), int(y)))

    populations = ([p.strip() for p in args.populations.split(",")]
                   if args.populations else None)

    with build_session(args) as session:
        plot_phase_trajectories(
            session,
            mnist_root        = args.mnist_root,
            n_images          = args.n_images,
            digits            = digits,
            n_warmup_steps    = args.n_warmup_steps,
            n_image_steps     = args.n_image_steps,
            n_shutdown_steps  = args.n_shutdown_steps,
            n_cycles          = args.n_cycles,
            probe_xy          = probe_xy,
            rate_transform    = args.rate_transform,
            populations       = populations,
            color_by          = args.color_by,
            normalize         = args.normalize,
            reset_mode        = args.reset_mode,
            output_dir        = args.output_dir,
        )