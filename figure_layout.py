"""
figure_layout.py
================
Figure creation, subplot grid management, font scaling, and file output.

Design principles
-----------------
- FigureLayout owns figure creation, subplot grid, font scaling,
  shared colorbars, and file output.
- All composite functions accept an optional FigureLayout for caller control,
  or create one automatically with sensible defaults.
- Can be used as a context manager.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Default subplot sizes (width_inches, height_inches)
# ============================================================

SUBPLOT_SIZE_HEATMAP    = (3.0, 2.8)   # roughly square for spatial maps
SUBPLOT_SIZE_LINEPLOT   = (4.5, 2.5)   # wider than tall for timeseries
SUBPLOT_SIZE_HISTOGRAM  = (3.5, 2.5)   # slight landscape for distributions
SUBPLOT_SIZE_SCATTER_3D = (4.0, 4.0)   # square for 3D projections


# ============================================================
# Style helpers
# ============================================================

def make_style(base=10):
    """
    Build a consistent matplotlib rcParams dict from a single base font size.
    All sizes are expressed as ratios of base so that changing base
    rescales everything proportionally.

    Args:
        base:  Base font size in points. Derive automatically with
               auto_base_fontsize() from subplot physical dimensions.

    Returns:
        dict suitable for plt.rc_context().
    """
    return {
        "font.size":          base,
        "axes.titlesize":     base,
        "axes.labelsize":     base * 0.9,
        "xtick.labelsize":    base * 0.8,
        "ytick.labelsize":    base * 0.8,
        "legend.fontsize":    base * 0.85,
        "figure.titlesize":   base * 1.1,
        "axes.titlepad":      base * 0.4,
        "axes.labelpad":      base * 0.3,
        "lines.linewidth":    1.2,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    }


def auto_base_fontsize(subplot_w_inches, subplot_h_inches,
                        min_size=6., max_size=12.):
    """
    Derive a base font size from the physical size of one subplot cell.
    Uses the shorter dimension as the binding constraint.

    Empirical linear fit:
        2 in -> ~8 pt  (cramped, smallest readable)
        3 in -> ~9.7 pt (typical heatmap)
        4 in -> ~11 pt (comfortable line plot)
        5 in -> ~12 pt (large / poster)

    Args:
        subplot_w_inches:  Width of one subplot cell in inches.
        subplot_h_inches:  Height of one subplot cell in inches.
        min_size:          Minimum font size in points.
        max_size:          Maximum font size in points.

    Returns:
        Float font size in points.
    """
    scale = min(subplot_w_inches, subplot_h_inches)
    base  = 5.5 + scale * 1.4
    return float(np.clip(base, min_size, max_size))


def figure_size(nrows, ncols, subplot_size):
    """
    Total figure size from grid dimensions and per-subplot size.

    Args:
        nrows, ncols:  Grid dimensions.
        subplot_size:  (width_inches, height_inches) per subplot cell.

    Returns:
        (total_width, total_height) in inches.
    """
    return (ncols * subplot_size[0], nrows * subplot_size[1])


def grid_shape(n, aspect="square"):
    """
    Calculate (nrows, ncols) for n subplots.

    Args:
        n:       Number of subplots.
        aspect:  Layout preference:
                 "square" -> as close to square as possible (default)
                 "row"    -> single row
                 "col"    -> single column
                 "wide"   -> at most 2 rows, favour columns

    Returns:
        (nrows, ncols) tuple.
    """
    if n == 1:
        return 1, 1
    if aspect == "row":
        return 1, n
    elif aspect == "col":
        return n, 1
    elif aspect == "wide":
        nrows = min(2, n)
        ncols = int(np.ceil(n / nrows))
        return nrows, ncols
    else:  # square
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        return nrows, ncols


# ============================================================
# FigureLayout
# ============================================================

class FigureLayout:
    """
    Central manager for figure creation, subplot grid, font scaling,
    shared colorbar, and file output.

    All composite plot functions accept an optional FigureLayout.
    If none is provided they create one internally with sensible defaults.

    Usage — automatic:
        plot_weight_map_grid(data)          # FigureLayout created internally

    Usage — caller controls layout:
        layout = FigureLayout(4, 4, subplot_size=(2.5, 2.5),
                              shared_colorbar=True)
        plot_weight_map_grid(data, layout=layout)

    Usage — as context manager:
        with FigureLayout(1, 1, subplot_size=SUBPLOT_SIZE_HEATMAP) as layout:
            animate_heatmap(data, layout=layout)
    """

    def __init__(self, nrows=1, ncols=1,
                 subplot_size=SUBPLOT_SIZE_HEATMAP,
                 shared_colorbar=False,
                 is_3d=False,
                 style=None):
        """
        Args:
            nrows, ncols:     Grid dimensions.
            subplot_size:     (width, height) inches per subplot cell.
            shared_colorbar:  Reserve a narrow axis on the right for a
                              single colorbar shared across all subplots.
            is_3d:            Create 3D projection axes.
            style:            Dict of rcParam overrides applied on top of
                              the auto-derived style.
        """
        self.nrows           = nrows
        self.ncols           = ncols
        self.shared_colorbar = shared_colorbar
        self.is_3d           = is_3d
        self.cbar_ax         = None
        self._ctx            = None

        base      = auto_base_fontsize(subplot_size[0], subplot_size[1])
        ctx_style = make_style(base)
        if style:
            ctx_style.update(style)

        self._ctx = plt.rc_context(ctx_style)
        self._ctx.__enter__()

        figsize = figure_size(nrows, ncols, subplot_size)

        if shared_colorbar:
            self.fig = plt.figure(figsize=figsize, layout="constrained")
            gs = self.fig.add_gridspec(
                nrows, ncols + 1,
                width_ratios=[1.0] * ncols + [0.05],
            )
            self.axes = np.array([
                [self.fig.add_subplot(
                    gs[r, c],
                    projection="3d" if is_3d else None)
                 for c in range(ncols)]
                for r in range(nrows)
            ])
            self.cbar_ax = self.fig.add_subplot(gs[:, -1])
        else:
            subplot_kw = {"projection": "3d"} if is_3d else {}
            self.fig, axes = plt.subplots(
                nrows, ncols,
                figsize=figsize,
                layout="constrained",
                subplot_kw=subplot_kw,
            )
            self.axes = np.array(axes).reshape(nrows, ncols)

    def flat_axes(self):
        """Flattened axes array for indexed iteration."""
        return self.axes.reshape(-1)

    def add_shared_colorbar(self, im, label=None):
        """
        Attach a colorbar to the reserved colorbar axis.

        Args:
            im:     Mappable returned by imshow or equivalent.
            label:  Colorbar label string.
        """
        if self.cbar_ax is not None:
            self.fig.colorbar(im, cax=self.cbar_ax, label=label or "")

    def add_per_axis_colorbar(self, im, ax, label=None):
        """Attach a colorbar beside a specific axes."""
        self.fig.colorbar(im, ax=ax, label=label or "", shrink=0.9)

    def save_or_show(self, output_path=None, dpi=150):
        """
        Save figure to file or display interactively.
        Format inferred from extension:
            .pdf -> vector, best for LaTeX
            .svg -> vector, best for web / editing
            .png -> raster at specified dpi
        If output_path is None, calls plt.show().
        Always closes the figure and exits the style context.
        """
        if output_path is None:
            plt.show()
        else:
            ext = os.path.splitext(output_path)[1].lower()
            if ext in (".pdf", ".svg"):
                self.fig.savefig(output_path, bbox_inches="tight")
            else:
                self.fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(self.fig)
        self._exit_context()

    def _exit_context(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._exit_context()

# Add this to figure_layout.py at the bottom, before the class definition

def _save_or_show_animation(anim, fig, output_path=None, fps=20, dpi=150):
    """
    Save a FuncAnimation to file or display interactively.
    
    Format inferred from extension:
        .mp4 -> H.264 via FFMpegWriter (requires ffmpeg)
        .gif -> animated GIF via PillowWriter
        .pdf -> multi-page PDF, one page per frame
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter
    
    if output_path is None:
        plt.show()
        return
    
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp4":
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(output_path, writer=writer, dpi=dpi)
    elif ext == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    elif ext == ".pdf":
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output_path) as pdf:
            for i in range(anim.save_count):
                anim._draw_frame(i)
                pdf.savefig(fig, bbox_inches="tight")
    else:
        raise ValueError(
            f"Unsupported animation format '{ext}'. Use .mp4, .gif, or .pdf"
        )
    plt.close(fig)