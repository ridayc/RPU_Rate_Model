"""
spec.py
=======
Declarative visualization specifications.
"""

from dataclasses import dataclass, field
from typing import Literal, Callable, Optional
import numpy as np


@dataclass
class VizSpec:
    """
    Declare how to visualize a given data type once, reuse everywhere.
    
    Examples:
        # Auto-detect everything
        spec = VizSpec()
        
        # Heatmap with explicit settings
        heatmap_spec = VizSpec(
            plot_type="heatmap",
            colormap="RdBu",
            normalize="global_symmetric"
        )
        
        # Distribution with KDE instead of histogram
        dist_spec = VizSpec(
            plot_type="kde",
            bw_method="silverman"
        )
    """
    
    # What kind of plot
    plot_type: Optional[Literal["heatmap", "histogram", "kde", "trajectory_3d", "timeseries"]] = None
    
    # Color & scaling
    colormap: Optional[str] = None          # None = auto (viridis for pos, RdBu for signed)
    normalize: Literal["frame", "global", "global_symmetric", "percentile"] = "frame"
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # time values for timeseries and animations
    time_values: Optional[np.ndarray] = None
    
    # Layout
    layout: Literal["auto", "grid", "single"] = "auto"
    aspect: Literal["square", "wide", "row", "col"] = "square"
    
    # Labels (callables receive index for multi-subplot)
    title: Optional[Callable[[int], str] | str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    
    # Data-specific
    bins: int | str = "auto"               # "auto" uses reasonable default
    bw_method: str | float = "scott"       # for KDE
    crop: Optional[str | dict] = None      # "bbox", "nonzero", or dict with x_min/x_max/y_min/y_max
    # timeseries std bands
    std_data: Optional[np.ndarray] = None  #std values for band
    band_alpha: float = 0.25
    band_color: Optional[str] = None  # None = use same as line
    
    # Output
    fps: int = 20
    dpi: int = 150
    
    def infer_from_data(self, data: np.ndarray) -> "VizSpec":
        """Return a copy with unset fields filled based on data shape and values."""
        import copy
        
        spec = copy.deepcopy(self)
        data = np.asarray(data)
        
        # Infer plot_type from data shape if not specified
        if spec.plot_type is None:
            if data.ndim >= 3:
                spec.plot_type = "heatmap"
            elif data.ndim == 2:
                if data.shape[1] == 3:
                    spec.plot_type = "trajectory_3d"
                else:
                    spec.plot_type = "histogram"
            else:
                spec.plot_type = "histogram"
        
        # Infer colormap from data sign if not specified
        if spec.colormap is None:
            if spec.plot_type == "heatmap":
                # Check if data has both positive and negative values
                if np.any(data < 0) and np.any(data > 0):
                    spec.colormap = "RdBu_r"
                else:
                    spec.colormap = "viridis"
            else:
                spec.colormap = "viridis"
        
        # Compute vmin/vmax if needed
        if spec.vmin is None or spec.vmax is None:
            if spec.normalize == "frame" and data.ndim >= 3 and data.shape[0] > 1:
                # For animation with per-frame normalization, we'll compute per frame
                # Don't set global bounds
                pass
            elif spec.normalize == "global_symmetric":
                # Symmetric around zero based on absolute max
                abs_max = np.abs(data).max()
                spec.vmin = -abs_max
                spec.vmax = abs_max
            elif spec.normalize == "percentile":
                # Use 1st and 99th percentile to ignore outliers
                spec.vmin = np.percentile(data, 1)
                spec.vmax = np.percentile(data, 99)
            else:  # "global" or other
                spec.vmin = float(data.min())
                spec.vmax = float(data.max())
        
        return spec