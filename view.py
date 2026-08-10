"""
view.py
=======
Single entry point for all visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from figure_layout import (
    FigureLayout,
    SUBPLOT_SIZE_HEATMAP,
    SUBPLOT_SIZE_HISTOGRAM,
    SUBPLOT_SIZE_SCATTER_3D,
    SUBPLOT_SIZE_LINEPLOT,
    grid_shape,
    _save_or_show_animation,
)
from plot_primitives import plot_heatmap, plot_histogram, plot_kde, plot_trajectory_3d, plot_timeseries, plot_timeseries_with_band
from spec import VizSpec


def view(data, spec=None, output_path=None, **overrides):
    data = np.asarray(data)
    if spec is None:
        spec = VizSpec()
    for key, value in overrides.items():
        if hasattr(spec, key):
            setattr(spec, key, value)
    spec = spec.infer_from_data(data)

    # Dispatch
    if spec.plot_type == "heatmap":
        return _view_heatmap(data, spec, output_path)
    elif spec.plot_type in ("histogram", "kde"):
        return _view_distribution(data, spec, output_path)
    elif spec.plot_type == "trajectory_3d":
        return _view_trajectory(data, spec, output_path)
    elif spec.plot_type == "timeseries":
        return _view_timeseries(data, spec, output_path)
    else:
        raise ValueError(f"Unsupported plot_type: {spec.plot_type}")


# ----------------------------------------------------------------------
# Shared helpers for static / animation / layout
# ----------------------------------------------------------------------

def _apply_title(ax, spec, idx=None):
    """Set title on axes using spec.title (string or callable with index)."""
    if spec.title is None:
        return
    if callable(spec.title):
        ax.set_title(spec.title(idx))
    else:
        ax.set_title(spec.title)


def _create_layout(nrows, ncols, subplot_size, shared_colorbar, is_3d, spec):
    """Create FigureLayout with proper defaults."""
    return FigureLayout(
        nrows, ncols,
        subplot_size=subplot_size,
        shared_colorbar=shared_colorbar,
        is_3d=is_3d,
        style=getattr(spec, 'style', None)
    )


def _run_animation(layout, update_func, n_frames, spec, output_path):
    """Create and save animation from update_func(frame_idx)."""
    anim = FuncAnimation(
        layout.fig, update_func,
        frames=n_frames,
        interval=max(20, int(1000 / spec.fps)),
        save_count=n_frames,
    )
    _save_or_show_animation(anim, layout.fig, output_path, fps=spec.fps, dpi=spec.dpi)
    layout._exit_context()
    return anim

def _compute_global_crop(data, spec):
    if spec.crop is None:
        return slice(None), slice(None)
    if isinstance(spec.crop, dict):
        rmin = spec.crop.get("y_min", 0)
        rmax = spec.crop.get("y_max", data.shape[-2] - 1)
        cmin = spec.crop.get("x_min", 0)
        cmax = spec.crop.get("x_max", data.shape[-1] - 1)
        return slice(rmin, rmax + 1), slice(cmin, cmax + 1)
    if spec.crop == "nonzero":
        # Find rows and cols where any positive value exists
        # Reshape to (..., H, W) and collapse all leading dims
        # Use np.any over all leading dimensions
        pos_mask = (data != 0)
        # Collapse all dimensions except the last two
        while pos_mask.ndim > 2:
            pos_mask = np.any(pos_mask, axis=0)
        # pos_mask is now 2D (H, W)
        if not np.any(pos_mask):
            return slice(None), slice(None)
        rows = np.any(pos_mask, axis=1)
        cols = np.any(pos_mask, axis=0)
        rmin = np.where(rows)[0][0]
        rmax = np.where(rows)[0][-1]
        cmin = np.where(cols)[0][0]
        cmax = np.where(cols)[0][-1]
        return slice(rmin, rmax + 1), slice(cmin, cmax + 1)
    return slice(None), slice(None)


# ----------------------------------------------------------------------
# Heatmap (static or animated)
# ----------------------------------------------------------------------

def _view_heatmap(data, spec, output_path):
    if data.ndim == 2:
        return _static_single_heatmap(data, spec, output_path)
    elif data.ndim == 3:
        # (Z, H, W) or (T, H, W)
        if data.shape[0] > 1 and output_path and output_path.endswith((".mp4", ".gif")):
            return _animate_single_heatmap(data, spec, output_path)
        else:
            return _static_heatmap_grid(data, spec, output_path)
    elif data.ndim == 4:
        return _animate_heatmap_grid(data, spec, output_path)
    else:
        raise ValueError(f"Unsupported heatmap shape: {data.shape}")


def _static_single_heatmap(data, spec, output_path):
    layout = _create_layout(1, 1, SUBPLOT_SIZE_HEATMAP, True, False, spec)
    ax = layout.flat_axes()[0]
    xs, ys = _compute_global_crop(data, spec)
    plot_data = data[xs, ys] if xs != slice(None) else data
    im = plot_heatmap(ax, plot_data, vmin=spec.vmin, vmax=spec.vmax,
                      cmap=spec.colormap,
                      title=spec.title if not callable(spec.title) else None,
                      xlabel=spec.xlabel, ylabel=spec.ylabel)
    layout.add_shared_colorbar(im)
    layout.save_or_show(output_path, dpi=spec.dpi)
    return layout


def _static_heatmap_grid(data, spec, output_path):
    n_plots = data.shape[0]
    nrows, ncols = grid_shape(n_plots, aspect=spec.aspect)
    layout = _create_layout(nrows, ncols, SUBPLOT_SIZE_HEATMAP, True, False, spec)
    axes = layout.flat_axes()
    # Compute global crop once using the full data (all subplots)
    xs, ys = _compute_global_crop(data, spec)
    im = None
    for i in range(n_plots):
        plot_data = data[i][xs, ys] if xs != slice(None) else data[i]
        title = None
        if spec.title:
            title = spec.title(i) if callable(spec.title) else spec.title
        elif n_plots > 1:
            title = f"z={i}"
        im = plot_heatmap(axes[i], plot_data, vmin=spec.vmin, vmax=spec.vmax,
                          cmap=spec.colormap, title=title,
                          xlabel=spec.xlabel, ylabel=spec.ylabel)
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    if im:
        layout.add_shared_colorbar(im)
    layout.save_or_show(output_path, dpi=spec.dpi)
    return layout


def _animate_single_heatmap(data, spec, output_path):
    T, H, W = data.shape
    layout = _create_layout(1, 1, SUBPLOT_SIZE_HEATMAP, True, False, spec)
    ax = layout.flat_axes()[0]
    xs, ys = _compute_global_crop(data, spec)

    def update(frame_idx):
        ax.cla()
        plot_data = data[frame_idx][xs, ys] if xs != slice(None) else data[frame_idx]
        im = plot_heatmap(ax, plot_data, vmin=spec.vmin, vmax=spec.vmax,
                          cmap=spec.colormap, xlabel=spec.xlabel, ylabel=spec.ylabel)
        layout.add_shared_colorbar(im)
        _apply_title(ax, spec, frame_idx)

    return _run_animation(layout, update, T, spec, output_path)


def _animate_heatmap_grid(data, spec, output_path):
    T, Z, H, W = data.shape
    nrows, ncols = grid_shape(Z, aspect=spec.aspect)
    layout = _create_layout(nrows, ncols, SUBPLOT_SIZE_HEATMAP, True, False, spec)
    axes = layout.flat_axes()
    for i in range(Z, len(axes)):
        axes[i].set_visible(False)

    xs, ys = _compute_global_crop(data, spec)

    def update(frame_idx):
        for ax in axes:
            ax.cla()
        frame = data[frame_idx]
        for zi in range(Z):
            plot_data = frame[zi][xs, ys] if xs != slice(None) else frame[zi]
            title = None
            if spec.title:
                title = spec.title(zi) if callable(spec.title) else spec.title
            elif Z > 1:
                title = f"z={zi}"
            im = plot_heatmap(axes[zi], plot_data, vmin=spec.vmin, vmax=spec.vmax,
                              cmap=spec.colormap, title=title,
                              xlabel=spec.xlabel, ylabel=spec.ylabel)
        layout.add_shared_colorbar(im)
        # Time display (optional)
        if spec.time_values is not None:
            if np.isscalar(spec.time_values):
                time_str = f"time = {spec.time_values}"
            else:
                time_str = f"time = {spec.time_values[frame_idx]}"
        else:
            time_str = f"frame {frame_idx}"
        layout.fig.suptitle(time_str)

    return _run_animation(layout, update, T, spec, output_path)


# ----------------------------------------------------------------------
# Distribution (histogram / KDE)
# ----------------------------------------------------------------------

def _view_distribution(data, spec, output_path):
    if data.ndim == 1:
        return _static_distribution(data, spec, output_path)
    elif data.ndim == 2:
        return _animate_distribution(data, spec, output_path)
    else:
        raise ValueError(f"Distribution data must be 1D or 2D, got {data.shape}")


def _static_distribution(data, spec, output_path):
    layout = _create_layout(1, 1, SUBPLOT_SIZE_HISTOGRAM, False, False, spec)
    ax = layout.flat_axes()[0]
    if spec.plot_type == "histogram":
        plot_histogram(ax, data, bins=spec.bins, density=True)
    else:
        plot_kde(ax, data, bw_method=spec.bw_method, fill=True)
    _apply_title(ax, spec, 0)
    layout.save_or_show(output_path, dpi=spec.dpi)
    return layout


def _animate_distribution(data, spec, output_path):
    T, N = data.shape
    layout = _create_layout(1, 1, SUBPLOT_SIZE_HISTOGRAM, False, False, spec)
    ax = layout.flat_axes()[0]

    # Precompute bins and global ylim
    if spec.plot_type == "histogram":
        global_min, global_max = data.min(), data.max()
        if spec.bins == "auto":
            bins = np.linspace(global_min, global_max, 50)
        elif isinstance(spec.bins, int):
            bins = np.linspace(global_min, global_max, spec.bins + 1)
        else:
            bins = spec.bins
        
        # Compute global max density/count across all frames
        max_count = 0
        for t in range(T):
            counts, _ = np.histogram(data[t].ravel(), bins=bins, density=True)
            max_count = max(max_count, counts.max())
        ylim = (0, max_count * 1.05)
    else:  # kde
        xlim = (data.min(), data.max())
        # For KDE, we need to evaluate density across a grid for each frame
        # Simpler: compute global max density by evaluating KDE on a fine grid
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(xlim[0], xlim[1], 200)
        max_density = 0
        for t in range(T):
            kde = gaussian_kde(data[t].ravel(), bw_method=spec.bw_method)
            density = kde.evaluate(x_grid)
            max_density = max(max_density, density.max())
        ylim = (0, max_density * 1.05)

    def update(frame_idx):
        ax.cla()
        frame = data[frame_idx]
        if spec.plot_type == "histogram":
            plot_histogram(ax, frame, bins=bins, density=True,
                           xlabel=spec.xlabel or "value",
                           ylabel=spec.ylabel or "density",
                           ylim=ylim)
        else:
            plot_kde(ax, frame, bw_method=spec.bw_method, fill=True,
                     xlim=xlim, ylim=ylim,
                     xlabel=spec.xlabel or "value",
                     ylabel=spec.ylabel or "density")
        _apply_title(ax, spec, frame_idx)

    return _run_animation(layout, update, T, spec, output_path)


# ----------------------------------------------------------------------
# 3D Trajectory
# ----------------------------------------------------------------------

def _view_trajectory(data, spec, output_path):
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Trajectory data must be (T, 3), got {data.shape}")
    T = data.shape[0]
    layout = _create_layout(1, 1, SUBPLOT_SIZE_SCATTER_3D, False, True, spec)
    ax = layout.flat_axes()[0]

    make_animation = (output_path and output_path.endswith((".mp4", ".gif")) and T > 1)
    if make_animation:
        pad = 0.1
        xlim = (data[:, 0].min() - pad, data[:, 0].max() + pad)
        ylim = (data[:, 1].min() - pad, data[:, 1].max() + pad)
        zlim = (data[:, 2].min() - pad, data[:, 2].max() + pad)

        def update(frame_idx):
            ax.cla()
            plot_trajectory_3d(ax, data[:frame_idx+1], color_by_time=True, cmap=spec.colormap)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            _apply_title(ax, spec, frame_idx)

        return _run_animation(layout, update, T, spec, output_path)
    else:
        plot_trajectory_3d(ax, data, color_by_time=True, cmap=spec.colormap)
        layout.save_or_show(output_path, dpi=spec.dpi)
        return layout


# ----------------------------------------------------------------------
# Timeseries
# ----------------------------------------------------------------------

def _view_timeseries(data, spec, output_path):
    data = np.asarray(data)
    
    # Normalise data to (n_lines, T)
    if data.ndim == 1:
        data_2d = data.reshape(1, -1)
        n_lines = 1
    else:
        if getattr(spec, 'transpose', False):
            data_2d = data.T
        elif data.shape[0] <= data.shape[1]:
            data_2d = data
        else:
            data_2d = data.T
        n_lines = data_2d.shape[0]
    T = data_2d.shape[1]

    # Time axis
    if spec.time_values is not None:
        if len(spec.time_values) != T:
            raise ValueError(f"time_values length {len(spec.time_values)} != T {T}")
        time_axis = spec.time_values
    else:
        time_axis = np.arange(T)

    # Check for std data
    use_band = (spec.std_data is not None)
    if use_band:
        std_data = np.asarray(spec.std_data)
        
        # If mean is 2D with one row and std is 1D, reshape std to (1, T)
        if data_2d.shape[0] == 1 and std_data.ndim == 1:
            std_data = std_data.reshape(1, -1)
    
        if std_data.shape != data_2d.shape:
            raise ValueError(f"std_data shape {std_data.shape} != data shape {data_2d.shape}")
            # For now, only support single line with band (can extend later)
        if n_lines > 1:
            # Option 1: raise
            raise NotImplementedError("Multiple lines with error bands not yet supported")
            # Option 2: loop over lines and call plot_timeseries_with_band per line
        # Use the band primitive
        layout = _create_layout(1, 1, SUBPLOT_SIZE_LINEPLOT, False, False, spec)
        ax = layout.flat_axes()[0]
        label = spec.labels[0] if hasattr(spec, 'labels') and spec.labels else None
        color = spec.band_color
        plot_timeseries_with_band(
            ax, time_axis, data_2d[0], std_data[0],
            label=label, color=color, alpha=spec.band_alpha,
            title=spec.title if not callable(spec.title) else None,
            xlabel="time", ylabel=spec.ylabel,
            ylim=getattr(spec, 'ylim', None)
        )
        layout.save_or_show(output_path, dpi=spec.dpi)
        return layout
    else:
        # Original single array logic (static or animated)
        layout = _create_layout(1, 1, SUBPLOT_SIZE_LINEPLOT, False, False, spec)
        ax = layout.flat_axes()[0]
        make_animation = (output_path and output_path.endswith((".mp4", ".gif")) and
                          T > 1 and getattr(spec, 'animate', False))
        if make_animation:
            ymin, ymax = data_2d.min(), data_2d.max()
            pad = (ymax - ymin) * 0.05
            def update(frame_idx):
                ax.cla()
                t = time_axis[:frame_idx+1]
                for row in data_2d:
                    ax.plot(t, row[:frame_idx+1])
                ax.set_xlim(time_axis[0], time_axis[-1])
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.set_xlabel("time")
                if spec.ylabel:
                    ax.set_ylabel(spec.ylabel)
                _apply_title(ax, spec, frame_idx)
            return _run_animation(layout, update, T, spec, output_path)
        else:
            for i, row in enumerate(data_2d):
                label = spec.labels[i] if hasattr(spec, 'labels') and spec.labels and i < len(spec.labels) else None
                ax.plot(time_axis, row, label=label)
            if hasattr(spec, 'labels') and spec.labels:
                ax.legend()
            ax.set_xlabel("time")
            if spec.ylabel:
                ax.set_ylabel(spec.ylabel)
            if hasattr(spec, 'ylim') and spec.ylim:
                ax.set_ylim(spec.ylim)
            _apply_title(ax, spec, 0)
            layout.save_or_show(output_path, dpi=spec.dpi)
            return layout