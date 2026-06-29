import numpy as np

from file_context import RateFileContext, StructureFileContext
from connectivity_reader import NetworkConnectivity
from data_view import DataView
from view import view
from spec import VizSpec

def receptive_field_plot(dv, pop_id, comp_id, x, y, t=None, chunks=None, fps=20):
    target_indices = dv.get_xy_stack(pop_id, x, y).ravel()  # (Z_target,)
    source_neurons = dv.get_source_indices(pop_id, comp_id, target_indices)  # (Z_target, k)
    weights = dv.get_input_weights(pop_id, comp_id, target_indices, t, chunks)  # (time, Z_target, k)
    
    # Handle 1D vs 3D case
    if weights.ndim == 2:
        # Add time dimension: (Z_target, k) -> (1, Z_target, k)
        weights = weights[np.newaxis, ...]
    n_time, Z_target, k = weights.shape
    pop2_id = dv.compartment_source_population(pop_id, comp_id)
    W, H, Z_source = dv.get_pop_size(pop2_id)
    
    projection = np.zeros((n_time, Z_target, W, H), dtype=np.float32)
    count = np.zeros((Z_target, W, H), dtype=np.float32)  # number of synapses per x,y location
    
    flat_source = source_neurons.ravel()  # (Z_target * k,)
    xs, ys, zs = dv.unravel_index(pop2_id,flat_source)
    xs = xs.reshape(Z_target, k)
    ys = ys.reshape(Z_target, k)
    
    for z_idx in range(Z_target):
        w = weights[:, z_idx, :]  # (time, k)
        np.add.at(projection, (slice(None), z_idx, xs[z_idx], ys[z_idx]), w)
        np.add.at(count, (z_idx, xs[z_idx], ys[z_idx]), 1)
    # avoid division by zero, then divide
    count = np.where(count > 0, count, 1)
    projection /= np.where(count > 0, count, 1)[np.newaxis, ...] #broadcast over time

    if t is None:
        t = dv._struct_ctx.timesteps
    else:
        t = dv._struct_ctx.timesteps[t]
    spec = VizSpec(
        plot_type="heatmap",
        time_values=t,
        crop="nonzero",
        fps=fps,
        normalize="global",
        title="heatmap",
    )
    if(projection.shape[0]>1):
        view(projection, spec = spec, output_path="animation.mp4")
    else:
        view(projection[0,:,:,:], spec = spec)



folder = "Storage"
#'''
rate_ctx = RateFileContext("rates/rates.h5")
struct_ctx = StructureFileContext("structure/structure.h5")
conn = NetworkConnectivity("connectivity/connectivity.h5")
#'''
'''
rate_ctx = RateFileContext(folder+"/rates.h5")
struct_ctx = StructureFileContext(folder+"/structure.h5")
conn = NetworkConnectivity(folder+"/connectivity.h5")
#'''
dv = DataView(rate_ctx=rate_ctx,struct_ctx=struct_ctx,connectivity=conn)
receptive_field_plot(dv,"E","P_E",14,15,t=236,chunks=2,fps=5)