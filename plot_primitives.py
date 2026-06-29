"""
plot_primitives.py
==================
Plotting primitives that render one frame onto an existing axes object.

Design principles
-----------------
- No figure creation
- No file saving
- No layout management
- Default to auto-scaling from immediate data (vmin=None, bins=None, etc.)
- Return the primary artist for optional colorbar attachment
- All functions accept an axes as the first argument
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


def plot_heatmap(ax, data, vmin=None, vmax=None, cmap="viridis",
                 title=None, xlabel=None, ylabel=None):
    """
    Render a 2D array as a heatmap.

    vmin and vmax default to None (auto-scale from data).
    Pass explicit values only when a fixed range is required
    (e.g. during animation via animate_plot).

    Args:
        ax:             Matplotlib axes.
        data:           2D array shape (H, W).
        vmin, vmax:     Color scale limits. None = auto from data.
        cmap:           Colormap name.
        title:          Axes title.
        xlabel, ylabel: Axis labels.

    Returns:
        AxesImage (mappable for colorbar).
    """
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect="equal", origin="lower",
                   interpolation="nearest")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=2)
    return im


def plot_timeseries(ax, t, values, labels=None, colors=None,
                    title=None, xlabel="time", ylabel=None,
                    ylim=None, legend=True):
    """
    Plot one or more timeseries lines.

    Args:
        ax:       Matplotlib axes.
        t:        1D array of timestep values (x axis).
        values:   2D array (n_lines, T) or 1D array (T,).
        labels:   Line labels for legend. None disables legend.
        colors:   Line colors. None uses default cycle.
        title:    Axes title.
        xlabel:   X axis label.
        ylabel:   Y axis label.
        ylim:     (ymin, ymax) or None for auto.
        legend:   Show legend when labels are provided.

    Returns:
        List of Line2D objects.
    """
    values = np.atleast_2d(values)
    lines = []
    for i, v in enumerate(values):
        kw = {}
        if labels is not None:
            kw["label"] = labels[i]
        if colors is not None:
            kw["color"] = colors[i]
        line, = ax.plot(t, v, **kw)
        lines.append(line)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if legend and labels:
        ax.legend(frameon=False)
    return lines


def plot_timeseries_with_band(ax, t, mean, std,
                               label=None, color=None,
                               title=None, xlabel="time", ylabel=None,
                               ylim=None, alpha=0.25):
    """
    Plot a timeseries mean line with a shaded std band.

    Args:
        ax:           Matplotlib axes.
        t:            1D timestep array.
        mean:         1D mean values array.
        std:          1D std values array.
        label:        Line label for legend.
        color:        Line and band color. None uses default cycle.
        alpha:        Opacity of the std band.
        ylim:         (ymin, ymax) or None for auto.

    Returns:
        (line, band) tuple of artists.
    """
    kw = {"color": color} if color else {}
    line, = ax.plot(t, mean, label=label, **kw)
    band = ax.fill_between(t, mean - std, mean + std,
                           alpha=alpha, **kw)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if label:
        ax.legend(frameon=False)
    return line, band


def plot_histogram(ax, data, bins=50, xlim=None, ylim=None,
                   color=None, title=None,
                   xlabel=None, ylabel="count",
                   density=False, log_y=False):
    """
    Plot a histogram of 1D or flat data.

    bins defaults to 50 integer bins (auto-range from data).
    Pass a bin edge array for fixed bins during animation.

    Args:
        ax:       Matplotlib axes.
        data:     Array of values (any shape, will be flattened).
        bins:     Int (number of bins, auto-range) or bin edge array
                  (use fixed edges from compute_animation_bounds when
                  animating to prevent frame-to-frame jumps).
        xlim:     (xmin, xmax) fixed axis limits or None for auto.
        ylim:     (ymin, ymax) fixed axis limits or None for auto.
        color:    Bar color. None uses default.
        density:  Normalise to density rather than counts.
        log_y:    Log scale on y axis.

    Returns:
        Tuple of (n, bins, patches) from ax.hist.
    """
    kw = {"color": color} if color else {}
    result = ax.hist(data.ravel(), bins=bins, density=density, **kw)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale("log")
    return result


def plot_scatter_3d(ax, xyz, color=None, cmap="viridis",
                    title=None,
                    xlabel="PC1", ylabel="PC2", zlabel="PC3",
                    s=8, alpha=0.7, elev=20, azim=45):
    """
    3D scatter plot on a 3D axes.

    Args:
        ax:    3D matplotlib axes (created with projection="3d").
        xyz:   Array shape (N, 3).
        color: Color array (N,) mapped through cmap, or a single color.
        s:     Marker size.
        alpha: Marker opacity.
        elev, azim: Initial viewing angle.

    Returns:
        PathCollection scatter artist.
    """
    if color is not None and np.ndim(color) > 0:
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        c=color, cmap=cmap, s=s, alpha=alpha)
    else:
        kw = {"color": color} if color else {}
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        s=s, alpha=alpha, **kw)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)
    return sc


def plot_trajectory_3d(ax, xyz, color_by_time=True, cmap="viridis",
                        title=None,
                        xlabel="PC1", ylabel="PC2", zlabel="PC3",
                        lw=1.2, alpha=0.8, elev=20, azim=45):
    """
    Connected 3D trajectory, optionally coloured by time position.

    Args:
        ax:             3D matplotlib axes.
        xyz:            Array shape (T, 3).
        color_by_time:  Colour segments from cmap so early/late trajectory
                        is visually distinguishable.
        lw:             Line width.
        alpha:          Line opacity.
        elev, azim:     Initial viewing angle.

    Returns:
        List of Line3D objects.
    """
    T = len(xyz)
    if color_by_time:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(T - 1, 1)))
        lines = []
        for i in range(T - 1):
            seg = xyz[i:i+2]
            ln, = ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                          color=colors[i], lw=lw, alpha=alpha)
            lines.append(ln)
    else:
        lines = [ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                         lw=lw, alpha=alpha)[0]]
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)
    return lines

def plot_kde(ax, data, bw_method="scott", xlim=None, ylim=None,
             color=None, fill=True, alpha=0.5, linewidth=1.5,
             title=None, xlabel=None, ylabel="density",
             log_x=False, log_y=False):
    """
    Plot a kernel density estimate of 1D data.

    Uses scipy.stats.gaussian_kde if available, falls back to matplotlib's
    hist(density=True) with many bins as a poor man's KDE.

    Args:
        ax:           Matplotlib axes.
        data:         Array of values (any shape, will be flattened).
        bw_method:    Bandwidth method for scipy KDE:
                      "scott" (default), "silverman", or float scalar.
                      Ignored if scipy not available.
        xlim:         (xmin, xmax) fixed axis limits or None for auto.
        ylim:         (ymin, ymax) fixed axis limits or None for auto.
        color:        Line and fill color. None uses default cycle.
        fill:         Fill area under the KDE curve.
        alpha:        Opacity for fill (0=transparent, 1=solid).
        linewidth:    Width of the KDE line.
        title:        Axes title.
        xlabel:       X axis label.
        ylabel:       Y axis label (default "density").
        log_x:        Log scale on x axis.
        log_y:        Log scale on y axis.

    Returns:
        Tuple of (line, fill_polygon) where fill_polygon is the
        Polygon artist (or None if fill=False).
    """
    data_flat = data.ravel()
    
    # Try to use scipy for proper KDE
    try:
        from scipy.stats import gaussian_kde
        
        kde = gaussian_kde(data_flat, bw_method=bw_method)
        
        # Determine x range for evaluation
        if xlim is not None:
            x_min, x_max = xlim
        else:
            # Use data range with 10% padding
            x_min, x_max = data_flat.min(), data_flat.max()
            pad = (x_max - x_min) * 0.1
            x_min, x_max = x_min - pad, x_max + pad
        
        # Evaluate KDE on fine grid
        x_grid = np.linspace(x_min, x_max, 200)
        y_vals = kde.evaluate(x_grid)
        
        # Plot
        kw = {"color": color} if color else {}
        line, = ax.plot(x_grid, y_vals, linewidth=linewidth, **kw)
        
        fill_poly = None
        if fill:
            fill_poly = ax.fill_between(x_grid, 0, y_vals, alpha=alpha, **kw)
        
    except ImportError:
        # Fallback: many-bin histogram as approximation
        import warnings
        warnings.warn(
            "scipy not available. Using histogram with 100 bins as KDE approximation. "
            "Install scipy for proper KDE smoothing.",
            UserWarning
        )
        
        kw = {"color": color, "histtype": "step", "density": True} if color else {"histtype": "step", "density": True}
        n, bins, patches = ax.hist(data_flat, bins=100, **kw)
        
        # For fallback, return line as None and patches as fill_poly
        line = None
        fill_poly = patches
        
        # Note: xlim auto-scaling handled by hist
    
    # Apply labels and limits
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    
    return line, fill_poly


def plot_kde_comparison(ax, data_list, labels=None, bw_method="scott",
                        colors=None, fill=False, alpha=0.5,
                        title=None, xlabel=None, ylabel="density",
                        xlim=None, ylim=None, legend=True):
    """
    Plot multiple KDEs on the same axes for comparison.

    Args:
        ax:           Matplotlib axes.
        data_list:    List of arrays, each flattened for KDE estimation.
        labels:       List of labels for legend.
        bw_method:    Bandwidth method passed to gaussian_kde.
        colors:       List of colors. None uses default cycle.
        fill:         Fill under curves (may become opaque/overlapping).
        alpha:        Opacity for fills (only relevant if fill=True).
        title:        Axes title.
        xlabel:       X axis label.
        ylabel:       Y axis label.
        xlim, ylim:   Axis limits (auto if None).
        legend:       Show legend if labels provided.

    Returns:
        List of (line, fill_polygon) tuples for each dataset.
    """
    results = []
    
    # Get default color cycle if colors not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    
    for i, data in enumerate(data_list):
        color = colors[i % len(colors)] if colors else None
        label = labels[i] if labels else None
        
        # Plot single KDE
        line, fill_poly = plot_kde(
            ax, data,
            bw_method=bw_method,
            color=color,
            fill=fill,
            alpha=alpha,
            title=title,  # Only applied once (will be overwritten, but that's fine)
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim
        )
        
        # Add label to line for legend
        if line is not None and label:
            line.set_label(label)
        
        results.append((line, fill_poly))
    
    # Apply legend (only once at the end)
    if legend and labels:
        ax.legend(frameon=False)
    
    # Ensure title and labels are set (may have been overwritten in loop)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    return results