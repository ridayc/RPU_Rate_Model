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
        if(field=="f" or field=="m" or field=="s"):
            f = dv.get_structure(pop_id,comp_id,"band_p_f",t)
            m = dv.get_structure(pop_id,comp_id,"band_p_m",t)
            s = dv.get_structure(pop_id,comp_id,"band_p_s",t)
            tot = f+m+s
            if(field=="f"):
                data = f/tot
            elif(field=="m"):
                data = m/tot
            else:
                data = s/tot
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

def weight_CV(x,y):
    x = x.reshape(*y.shape, -1)
    print(x.shape)
    return np.std(x,axis=1)/np.mean(x,axis=1)


def quick_log(x,y):
    return np.log(x+1e-8)

def quick_exp(x,y):
    return np.exp(x)

def id(x,y):
    return x

def default(x,y):
    return x if y is None else x/y

def threshold(x,y):
    return np.mean(x[..., None] > y.reshape(*x.shape, -1), axis=-1)

def true_weights(x,y):
    return x[...,None].repeat_interleave()*y.reshape(*x.shape,-1)

def local_function(x,axis=0,name=""):
    if name=="mean":
        return np.mean(x,axis=axis)
    elif name=="median":
        return np.median(x,axis=axis)
    elif name=="log":
        return np.log(x+1e-8)
    elif name=="gmean":
        return np.exp(np.mean(np.log(x),axis=axis))
    elif name=="exp":
        return np.exp(x) 
    elif name=="cv":
        return np.std(x,axis=axis)/(np.mean(x,axis=axis)+1e-9)
    else:
        return x



def structure_heatmap(dv, pop_id, title="", lfunc="", zmean=False, fps=5,**kwargs):
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
        data = local_function(data,name=lfunc)
        if zmean:
            data = np.mean(data,axis=0)
        view(data, spec = spec)

    else:
        local_function(data,name=lfunc)
        if zmean:
            data = np.mean(data,axis=1)
        view(data, spec = spec, output_path="animation.mp4")

def structure_histogram(dv, pop_id, title="", lfunc="", fps=5,hist_type="kde",**kwargs):
    data,t = data_loader(dv,pop_id,**kwargs)
    spec = VizSpec(
        plot_type=hist_type,
        time_values=t,
        fps=fps,
        normalize="global",
        title=f"t = {t}" if isinstance(t, (int, np.integer)) else lambda i: f"t = {t[i]}"
    )
    if isinstance(t, (int, np.integer)):
        view(local_function(data,name=lfunc), spec = spec)
    else:
        view(local_function(data,name=lfunc), spec = spec, output_path="animation.mp4")


def time_series(dv, pop_id,title="", lfunc="",**kwargs):
    data,t = data_loader(dv,pop_id,**kwargs)
    spec = VizSpec(
        plot_type="timeseries",
        #time_values=t,
        title=title,
        ylabel="Rates",
    )
    view(local_function(data,name=lfunc,axis=1),spec=spec)



rate_ctx = RateFileContext("rates/rates.h5")
struct_ctx = StructureFileContext("structure/structure.h5")
conn = NetworkConnectivity("connectivity/connectivity.h5")
dv = DataView(rate_ctx=rate_ctx,struct_ctx=struct_ctx,connectivity=conn)
nwritten = struct_ctx.n_written
sl = slice(1,nwritten)
print("Number of current steps: "+str(nwritten))
#time_series(dv,pop_id="E",comp_id="E_E",field="ravg",field2="",function=default,lfunc="gmean",t=sl)
#time_series(dv,pop_id="E",comp_id="E_E",field="w",field2="a",function=weight_CV,lfunc="mean",t=sl)
#structure_heatmap(dv,pop_id="E",comp_id="E_E",field="f",field2="",t=420,function=default,fps=5)
structure_histogram(dv,pop_id="I",comp_id="I_I",field="w",field2="",t=18,function=quick_log,fps=5,hist_type="histogram")
#structure_histogram(dv,pop_id="E",comp_id="E_E",field="w",field2="a",function=weight_CV,t=5,fps=5,hist_type="histogram")
#structure_heatmap(dv,"E",t=slice(100,1000),fps=5)