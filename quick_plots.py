import numpy as np

from file_context import RateFileContext, StructureFileContext
from connectivity_reader import NetworkConnectivity
from data_view import DataView
from view import view
from spec import VizSpec

def data_loader(dv,pop_id,comp_id=None,field=None,field2=None,t=None,function = lambda x,y: x if y is None else x/y):
    if(field is None):
        data = dv.get_rates(pop_id,t)
    else:
        data = dv.get_structure(pop_id,comp_id,field,t)
    data2 = None
    if(field2 is not None and field2!=""):
        data2 = dv.get_structure(pop_id,comp_id,field2,t)
    if(comp_id is None):
        if t is None:
            t = dv._rate_ctx.timesteps
        else:
            t = dv._rate_ctx.timesteps[t]
    else:
        if t is None:
            t = dv._struct_ctx.timesteps
        else:
            t = dv._struct_ctx.timesteps[t]
    return function(data,data2),t

def EMA_CV(x,y):
    return np.sqrt(np.maximum(y-x*x,0))/x

def quick_log(x,y):
    return np.log(x+1e-9)

def id(x,y):
    return x



def structure_heatmap(dv, pop_id, title="", zmean=False, fps=5,**kwargs):
    X,Y,Z = dv.get_pop_size(pop_id)
    data,t = data_loader(dv,pop_id,**kwargs)
    data = dv.reshape_spatial_data(pop_id,data)
    spec = VizSpec(
        plot_type="heatmap",
        time_values=t,
        fps=fps,
        normalize="global",
        title=title,
    )
    if isinstance(t, (int, np.integer)):
        if zmean:
            data = np.mean(data,axis=0)
        view(data, spec = spec)

    else:
        if zmean:
            data = np.mean(data,axis=1)
        view(data, spec = spec, output_path="animation.mp4")

def structure_histogram(dv, pop_id, title="", fps=5,hist_type="kde",**kwargs):
    data,t = data_loader(dv,pop_id,**kwargs)
    spec = VizSpec(
        plot_type=hist_type,
        time_values=t,
        fps=fps,
        normalize="global",
        title=f"t = {t}" if isinstance(t, (int, np.integer)) else lambda i: f"t = {t[i]}"
    )
    if isinstance(t, (int, np.integer)):
        view(data, spec = spec)

    else:
        view(data, spec = spec, output_path="animation.mp4")


def time_series(dv, pop_id,title="",**kwargs):
    data,t = data_loader(dv,pop_id,**kwargs)
    series = np.median(data,axis=1)
    spec = VizSpec(
        plot_type="timeseries",
        #time_values=t,
        title=title,
        ylabel="Rates",
    )
    view(series,spec=spec)



rate_ctx = RateFileContext("rates/rates.h5")
struct_ctx = StructureFileContext("structure/structure.h5")
conn = NetworkConnectivity("connectivity/connectivity.h5")
dv = DataView(rate_ctx=rate_ctx,struct_ctx=struct_ctx,connectivity=conn)
#structure_heatmap(dv,pop_id="E",comp_id="E_E",field="ravg",field2="r2avg",function=EMA_CV,t=45,fps=5,zmean=True)
structure_histogram(dv,pop_id="E",comp_id="E_E",field="w",field2="",function=quick_log,t=229,fps=5,hist_type="histogram")
nwritten = struct_ctx.n_written
sl = slice(0,nwritten)
#time_series(dv,pop_id="S",comp_id="E_S",field="ravg",field2="r2avg",function=EMA_CV,t=sl)
#structure_heatmap(dv,"E",t=slice(100,1000),fps=5)