import numpy as np, math
import matplotlib.pyplot as plt
import torch

# ==============================
# Generic volume / map plotting
# ==============================

def plot_volume(
    vol,
    mode="slice",
    z=0,
    func=lambda a: np.max(a, axis=2),
    vmin=None,
    vmax=None,
    normalize=False,
    title=None,
    figsize=None,
    tight=True,
    cmap="viridis",
    ax=None,
):
    """
    Plot a 3D array vol (W,H,Z) as:
      - mode='slice' : single depth slice z
      - mode='func'  : apply func(W,H,Z)->(W,H) (e.g., np.max(axis=2))
      - mode='grid'  : all slices in a grid
    """
    if torch.is_tensor(vol):
        vol = vol.detach().cpu().numpy()
    assert vol.ndim == 3, f"expected (W,H,Z), got {vol.shape}"
    W, H, Z = vol.shape

    def _imshow(img, ax_, title_):
        if normalize:
            mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
            data = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
            vmin_, vmax_ = 0.0, 1.0
        else:
            data = img
            if vmin is not None and vmax is not None:
                vmin_, vmax_ = vmin, vmax
            else:
                # Only compute from data if user didn't supply explicit limits
                if np.isnan(img).any():
                    vmin_, vmax_ = None, None
                else:
                    vmin_, vmax_ = float(np.min(img)), float(np.max(img))

        im = ax_.imshow(data.T, origin="lower", vmin=vmin_, vmax=vmax_, aspect="equal", cmap=cmap)
        ax_.set_xticks([])
        ax_.set_yticks([])
        if title_:
            ax_.set_title(title_, fontsize=10)
        return im

    # --- Single slice ---
    if mode == "slice":
        if not (0 <= z < Z):
            raise ValueError(f"z={z} out of range [0,{Z-1}]")
        fig = plt.figure(figsize=figsize or (4, 4)) if ax is None else ax.figure
        ax_ = fig.add_subplot(1, 1, 1) if ax is None else ax
        im = _imshow(vol[:, :, z], ax_, title)
        if ax is None:
            fig.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)
        if tight and ax is None:
            fig.tight_layout()
        return fig, ax_

    # --- Function over depth ---
    if mode == "func":
        img = func(vol)
        if img.shape != (W, H):
            raise ValueError(f"func must return (W,H), got {img.shape}")
        fig = plt.figure(figsize=figsize or (4, 4)) if ax is None else ax.figure
        ax_ = fig.add_subplot(1, 1, 1) if ax is None else ax
        im = _imshow(img, ax_, title)
        if ax is None:
            fig.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)
        if tight and ax is None:
            fig.tight_layout()
        return fig, ax_

    # --- Grid of slices ---
    if mode == "grid":
        cols = int(math.ceil(math.sqrt(Z)))
        rows = int(math.ceil(Z / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize or (3 * cols, 3 * rows))
        axes = np.array(axes).reshape(rows, cols)
        ims = []
        for zi in range(rows * cols):
            r, c = divmod(zi, cols)
            ax_ = axes[r, c]
            if zi < Z:
                im = _imshow(vol[:, :, zi], ax_, f"z={zi}")
                ims.append(im)
            else:
                ax_.axis("off")
        if ims:
            fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        if title:
            fig.suptitle(title, y=0.99)
        if tight:
            fig.tight_layout()
        return fig, axes

    raise ValueError("mode must be one of {'slice','func','grid'}")


def pop_volume(net, pop_id, field="rates"):
    """
    field: 'rates' (default) or any 1D per-neuron tensor of length W*H*Z attached to the pop.
    """
    pop = net.populations[pop_id]
    W, H, Z = pop.size
    x = getattr(pop, field)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x.reshape(W, H, Z)


def comp_volume_target(comp, field_1d):
    """
    Reshape a per-target 1D tensor to (W,H,Z) in the *target* geometry.
    """
    W, H, Z = comp.target.size
    x = field_1d
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x.reshape(W, H, Z)


def comp_band_phase(comp, band_key, side="out"):
    """
    Phase φ in radians for a band (if using frequency bands with cos/sin tracking).
    side='out' -> target geometry (cout/sout),
    side='in'  -> source geometry (cin/sin).
    """
    rb = comp.rate_band[band_key]
    if side == "out":
        W, H, Z = comp.target.size
        c = rb["cout"].detach().cpu().numpy().reshape(W, H, Z)
        s = rb["sout"].detach().cpu().numpy().reshape(W, H, Z)
    elif side == "in":
        W, H, Z = comp.source.size
        c = rb["cin"].detach().cpu().numpy().reshape(W, H, Z)
        s = rb["sin"].detach().cpu().numpy().reshape(W, H, Z)
    else:
        raise ValueError("side must be 'out' or 'in'")
    return np.arctan2(s, c)  # [-π, π]


def comp_band_amplitude(comp, band_key, side="out", normalize_by_avg=False):
    """
    Band amplitude proxy = sqrt(cos^2 + sin^2).
    If normalize_by_avg=True, divide by smoothed rate ('in'/'out').
    """
    rb = comp.rate_band[band_key]
    if side == "out":
        W, H, Z = comp.target.size
        c = rb["cout"]
        s = rb["sout"]
        avg = rb["out"] if normalize_by_avg else None
    elif side == "in":
        W, H, Z = comp.source.size
        c = rb["cin"]
        s = rb["sin"]
        avg = rb["in"] if normalize_by_avg else None
    else:
        raise ValueError("side must be 'out' or 'in'")

    amp = torch.sqrt(c * c + s * s)
    if normalize_by_avg:
        amp = amp / (avg + 1e-12)
    return amp.detach().cpu().numpy().reshape(W, H, Z)


# ============================
# Histograms and basic stats
# ============================

def plot_population_rate_hist(net, pop_id, bins=50, range=None, figsize=None, title=None):
    r = net.populations[pop_id].rates
    if torch.is_tensor(r):
        r = r.detach().to("cpu").numpy()
    fig = plt.figure(figsize=figsize or (4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(r, bins=bins, range=range)
    ax.set_xlabel("rate")
    ax.set_ylabel("count")
    ax.set_title(title or f"{pop_id} rate histogram")
    fig.tight_layout()
    return fig, ax


def plot_compartment_weight_hist(
    comp,
    bins=50,
    logx=False,
    logy=False,
    title=None,
    figsize=(4, 3),
    range=None,
):
    """
    Histogram of synaptic weights in a compartment (comp.w).
    """
    w = comp.w.detach().cpu().numpy()
    if logx:
        w = np.clip(w, 1e-12, None)
        vals = np.log10(w)
        xlabel = "log10(weight)"
    else:
        vals = w
        xlabel = "weight"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(vals, bins=bins, range=range, log=logy)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if title is None:
        title = f"{comp.target.id} ← {comp.source.id} weights"
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_compartment_amplitude_hist(
    comp,
    bins=50,
    logx=False,
    logy=False,
    title=None,
    figsize=(4, 3),
    range=None,
):
    """
    Histogram of per-target amplitudes a_i in a compartment (comp.a).
    """
    a = comp.a.detach().cpu().numpy()
    if logx:
        a = np.clip(a, 1e-12, None)
        vals = np.log10(a)
        xlabel = "log10(amplitude)"
    else:
        vals = a
        xlabel = "amplitude"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(vals, bins=bins, range=range, log=logy)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if title is None:
        title = f"{comp.target.id} ← {comp.source.id} amplitudes (a)"
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_compartment_amplitude_map(
    comp,
    mode="func",
    z=0,
    func=lambda vol: np.max(vol, axis=2),
    vmin=None,
    vmax=None,
    normalize=False,
    title=None,
    figsize=None,
    cmap="viridis",
    tight=True,
):
    """
    Plot per-target amplitudes 'a' as a (W,H,Z) volume using plot_volume.
    mode: 'slice' | 'func' | 'grid'
    """
    vol = comp_volume_target(comp, comp.a)
    if title is None:
        title = f"{comp.target.id} ← {comp.source.id} amplitude (a)"
    return plot_volume(
        vol,
        mode=mode,
        z=z,
        func=func,
        vmin=vmin,
        vmax=vmax,
        normalize=normalize,
        title=title,
        figsize=figsize,
        cmap=cmap,
        tight=tight,
    )


# ============================
# Receptive field utilities
# ============================

def index_to_xyz(index, size):
    """
    Convert flat neuron index -> (x,y,z) for a population with size [W,H,Z].

    Flattening in your network code is:
        idx = x*(H*Z) + y*Z + z
    so we invert that here.
    """
    W, H, Z = size
    idx = int(index)
    z = idx % Z
    tmp = idx // Z
    y = tmp % H
    x = tmp // H
    return x, y, z


def receptive_field_volume(comp, target_index, aggregate="sum"):
    """
    Build a (W_s, H_s, Z_s) volume of incoming weights for a single target neuron.

    - comp.w_ind[0,:] : target indices
    - comp.w_ind[1,:] : source indices (flattened in source geometry)
    - comp.w          : weights
    """
    W_s, H_s, Z_s = comp.source.size
    target_index = int(target_index)

    # mask for all synapses ending on this target neuron
    tgt = comp.w_ind[0]
    src = comp.w_ind[1]
    if torch.is_tensor(tgt):
        tgt = tgt.detach().cpu()
        src = src.detach().cpu()
    w = comp.w.detach().cpu()

    mask = (tgt == target_index)
    if not mask.any():
        # No afferents? Return all zeros.
        return np.zeros((W_s, H_s, Z_s), dtype=np.float32)

    src_idx = src[mask]
    w_vals = w[mask]

    vol = torch.zeros((W_s, H_s, Z_s), dtype=torch.float32)

    # Map each src index back to (x,y,z)
    z = src_idx % Z_s
    tmp = src_idx // Z_s
    y = tmp % H_s
    x = tmp // H_s

    if aggregate == "sum":
        # If multiple synapses to same voxel, sum them
        for xi, yi, zi, wi in zip(x, y, z, w_vals):
            vol[xi, yi, zi] += wi
    elif aggregate == "max":
        for xi, yi, zi, wi in zip(x, y, z, w_vals):
            vol[xi, yi, zi] = max(vol[xi, yi, zi], wi)
    else:
        raise ValueError("aggregate must be 'sum' or 'max'")

    return vol.numpy()


def _add_grid_overlay(ax, W, H, n_ticks=10):
    """
    Add a light grid and a few coordinate ticks to an RF axis.
    """
    n_ticks = min(n_ticks, max(W, H))
    if n_ticks <= 1:
        return

    xt = np.linspace(0, W - 1, n_ticks, dtype=int)
    yt = np.linspace(0, H - 1, n_ticks, dtype=int)

    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([str(v) for v in xt], fontsize=7)
    ax.set_yticklabels([str(v) for v in yt], fontsize=7)

    ax.grid(which="major", color="white", alpha=0.25, lw=0.5)


def plot_receptive_field(
    comp,
    target_index,
    mode="func",
    z=0,
    func=lambda v: v.max(axis=2),
    vmin=None,
    vmax=None,
    normalize=True,
    title=None,
    figsize=None,
    cmap="viridis",
    ax=None,
):
    """
    Plot receptive field for one target neuron in a compartment.

    Uses receptive_field_volume() and then hands off to plot_volume().
    Adds a fine grid & coordinate ticks on top (for RF inspection).
    """
    vol = receptive_field_volume(comp, target_index)

    # Call generic volume plotter (without tight layout so grid fits)
    fig, ax_ = plot_volume(
        vol,
        mode=mode,
        z=z,
        func=func,
        vmin=vmin,
        vmax=vmax,
        normalize=normalize,
        title=title,
        figsize=figsize,
        cmap=cmap,
        tight=False,
        ax=ax,
    )

    # Grid overlay (we know the spatial dims are W,H from vol.shape[:2])
    W_s, H_s, _ = vol.shape
    _add_grid_overlay(ax_, W_s, H_s, n_ticks=10)

    # Don't re-tight-layout here; let caller control if they have many subplots
    return fig, ax_


def plot_receptive_field_column(
    comp,
    x,
    y,
    mode="func",
    func=lambda v: v.max(axis=2),
    normalize=True,
    cmap="viridis",
    figsize=None,
):
    """
    Plot RFs for all neurons at same (x,y) across depth in target geometry.

    For each depth z_t, we compute the flat index and plot its RF.
    """
    W_t, H_t, Z_t = comp.target.size
    x = max(0, min(W_t - 1, x))
    y = max(0, min(H_t - 1, y))

    # All depths for this (x,y)
    indices = []
    for z_t in range(Z_t):
        idx = x * (H_t * Z_t) + y * Z_t + z_t
        indices.append(idx)

    n = len(indices)
    cols = min(n, 4)
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        xt, yt, zt = index_to_xyz(idx, comp.target.size)

        title = f"(x,y,z)=({xt},{yt},{zt})"
        plot_receptive_field(
            comp,
            target_index=idx,
            mode=mode,
            func=func,
            normalize=normalize,
            title=title,
            cmap=cmap,
            ax=ax,
        )

    # Turn off unused axes (if any)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    return fig, axes