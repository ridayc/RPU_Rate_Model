import torch
import math
import numpy as np
import copy as cp
import h5py

class Network:
    def __init__(self,device,populations_param,compartments_param):
        self.device = device
        self.time = 0
        self.populations = {}
        for v in populations_param.values():
            self.populations[v["id"]] = Population(v,self)
        for v in compartments_param.values():
            c = Compartment(v,self)
            c.target.add_compartment(c)

    def iterate(self):
        # 1. Calculate local crosscorrelation and temporal and spatial coefficients for different types of inhibitory machinary
        # 2. Update all compartment weights based on current pre-/post weights
        # 3. In a second run through update all population firing rates (using the updated weights)
        # 4. Increment timestep
        for v in self.populations.values():
            v.E_I_balance()
        for v in self.populations.values():
            v.update_weights()
        for v in self.populations.values():
            v.activations()
        for v in self.populations.values():
            v.update_rates()
        self.time+=1

    def iter_compartments(self):
        for p in self.populations.values():
            for c in p.compartments.values():
                yield c

    def save(self, path: str):
        # Full snapshot on whatever device it’s currently on
        torch.save(self, path, pickle_protocol=4)

    @staticmethod
    def load(path: str, device=None):
        # Remap *all* tensors to CPU/GPU on load (works for nested dicts too)
        if device is None:
            net = torch.load(path)  # original devices
        else:
            net = torch.load(path,map_location=device)
            net.device = device
        return net

class Population:
    def __init__(self,population_param,network):
        self.net = network
        self.compartments = {}
        self.id = population_param["id"]
        self.size = cp.deepcopy(population_param["size"])
        self.nneu = math.prod(self.size)
        self.tau = population_param["tau"]
        self.p = population_param["p"]
        self.r0 = population_param["r0"]
        self.baseline = population_param["baseline"]
        self.cap = population_param["cap"]
        self.activation = population_param["activation"]
        self.rates = torch.zeros(self.nneu).to(self.net.device)
        self.uact = torch.zeros(self.nneu).to(self.net.device)
        self.ravg = self.r0
        self.rsq = self.r0*self.r0
        self.band = cp.deepcopy(compartment_param["band"])
        self.rate_band = {}
        for k in self.band:
            self.rate_band[k] = {"mu": torch.full((self.nneu,),self.r0,device=self.net.device), "r2": torch.full((self.nneu,),self.r0*self.r0,device=self.net.device)}

    def add_compartment(self,compartment):
        self.compartments[compartment.id] = compartment

    def activations(self):
        u = {}
        for v in self.compartments.values():
            v.local_rate()
            u[v.id] = v.lrates
        if len(u)>0:
            self.uact[:] = torch.clamp(torch.clamp(self.activation(u),min=0)**self.p*self.r0**(1-self.p),min=self.baseline,max=self.cap)
        else:
            self.uact[:] = torch.clamp(torch.zeros(self.nneu).to(self.net.device),min=0)

    def update_rates(self):
        self.rates[:] = (1-self.tau)*self.rates+self.tau*self.uact
        self.ravg = torch.mean(self.rates).item()
        self.rsq = torch.mean(self.rates*self.rates).item()
        for k,v in self.rate_band.items():
            self.rate_band[k]["mu"] = smoothing(self.rate_band[k]["mu"],self.rates,self.band[k]["tau"])

    def update_weights(self):
        for v in self.compartments.values():
            v.update_weights()

    def E_I_balance(self):
        counter = 0
        # get
        for v in self.compartments.values():
            if (v.stype=="E"):
                v.mixing_update()
        for v in self.compartments.values():
            if (v.stype=="PV"):
                v.pv_pooling()
                v.pv_band_power()


# connectivity modules connecting two neuron populations. Two populations can have multiple separate compartments together.
class Compartment:
    def __init__(self,compartment_param,network):
        self.targetid = compartment_param["target"]
        self.sourceid = compartment_param["source"]
        self.target = network.populations[self.targetid]
        self.source = network.populations[self.sourceid]
        self.net = network
        self.id = compartment_param["id"]
        self.A = compartment_param["A"]
        self.A0 = compartment_param["A0"]
        self.lA0 = np.log(np.abs(compartment_param["A0"]))
        self.type = np.sign(self.A)
        self.A0 = np.sign(self.A0)*self.A0
        self.A*=self.type
        self.eta = compartment_param["eta"]
        self.nu = compartment_param["nu"]
        self.beta = compartment_param["beta"]
        self.delta = compartment_param["delta"]
        self.rho = compartment_param["rho"]
        self.tau = compartment_param["tau"]
        self.rin = compartment_param["rin"]
        self.rout = compartment_param["rout"]
        self.tauin = compartment_param["tauin"]
        self.tauout = compartment_param["tauout"]
        self.eps = compartment_param["eps"]
        self.stype = compartment_param["stype"]
        self.M = 0
        self.rate_target = compartment_param["rate_target"]
        # measurment variables for excitatory compartments. Local correlation and coefficient of variance markers
        if(self.stype=="E"):
            self.tauf = compartment_param["tauf"]
            self.taus = compartment_param["taus"]
            self.rit = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
            self.rjt = torch.full((self.source.nneu,),self.rate_target).to(self.net.device)
            self.ri2t = torch.full((self.target.nneu,),self.rate_target*self.rate_target).to(self.net.device)
            self.rj2t = torch.full((self.source.nneu,),self.rate_target*self.rate_target).to(self.net.device)
            self.rs = torch.zeros(self.target.nneu).to(self.net.device)
            self.r2s = torch.zeros(self.source.nneu).to(self.net.device)
            self.sigi = torch.zeros(self.target.nneu).to(self.net.device)
            self.sigj = torch.zeros(self.source.nneu).to(self.net.device)
            # local covariance tracker
            self.H = torch.zeros(self.target.nneu).to(self.net.device)
        elif(self.stype=="PV"):
            self.Pf = torch.full((self.target.nneu,),0.).to(self.net.device)
            self.Pm = torch.full((self.target.nneu,),0.).to(self.net.device)
            self.Ps = torch.full((self.target.nneu,),0.).to(self.net.device)
            self.eta_f = compartment_param["eta_f"]
            self.eta_s = compartment_param["eta_s"]
            self.theta_f = compartment_param["theta_f"]
            self.theta_s = compartment_param["theta_s"]
        # local correlation
        self.C_fast = torch.zeros(self.target.nneu).to(self.net.device)
        self.C = torch.zeros(self.target.nneu).to(self.net.device)
        self.C2 = torch.zeros(self.target.nneu).to(self.net.device)
        # local temporal coefficient of variance for individual neurons
        self.CVt_fast = torch.zeros(self.target.nneu).to(self.net.device)
        self.CVt = torch.zeros(self.target.nneu).to(self.net.device)
        # local population coefficient of variance estimate from individual neuron inputs
        self.CVs_fast = torch.zeros(self.target.nneu).to(self.net.device)
        self.CVs = torch.zeros(self.target.nneu).to(self.net.device)
        self.mixing = torch.zeros(self.target.nneu).to(self.net.device)

        self.rate_average = torch.zeros(self.target.nneu).to(self.net.device)
        self.rate_average+=self.rate_target
        self.rate_square = torch.full((self.target.nneu,),self.rate_target*self.rate_target).to(self.net.device)
        self.rate_in = torch.full((self.source.nneu,),self.rate_target).to(self.net.device)
        self.rj2 = torch.full((self.target.nneu,),0).to(self.net.device)
        self.rate_out = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        self.band = cp.deepcopy(compartment_param["band"])
        self.rate_band = {}
        # create pytorch tensors to store exponential sliding windows for cosine and sine frequency bands
        for k in self.band:
            self.rate_band[k] = {"mu": torch.full((self.target.nneu,),self.self.rate_target,device=self.net.device), "r2": torch.full((self.target.nneu,),self.self.rate_target*self.self.rate_target,device=self.net.device)}
        target_size = torch.Size(self.target.size)
        origin_size = torch.Size(self.source.size)
        points,self.k = sample_synapses(origin_size,target_size,compartment_param["ellipse"][0],compartment_param["ellipse"][0],math.prod(self.target.size),compartment_param["tsyn"])
        self.ones = torch.full((self.source.nneu,),1.).to(self.net.device)
        self.nsyn = self.target.nneu*self.k
        self.inds = torch.arange(self.target.nneu).repeat_interleave(self.k).to(self.net.device)
        self.indt = points.view(-1).to(self.net.device)
        self.w_ind = torch.stack((self.inds,self.indt))
        self.w = torch.zeros(self.nsyn).to(self.net.device)
        #self.w+= 1./self.k
        self.w[:] = torch.exp(self.eps*torch.randn_like(self.w))
        self.a = torch.zeros(self.target.nneu).to(self.net.device)
        self.a+=self.A
        self.loga = torch.zeros(self.target.nneu).to(self.net.device)
        self.loga+=np.log(self.A)
        self.lrates = torch.zeros(self.target.nneu).to(self.net.device)
        self.W = torch.sparse_coo_tensor(self.w_ind,self.w,size=(self.target.nneu, self.source.nneu)).coalesce().to(self.net.device)
        self.w_ind[0,:] = self.W.indices()[0,:]
        self.w_ind[1,:] = self.W.indices()[1,:]
        self.w[:] = self.W.values()
        self.normalize_weights()
        self.W._values()[:] = self.w
        # not needed yet, but maybe later if we have non-homogenous starting weights


    def local_rate(self):
        '''
        Compute this compartment's contribution to the target population:
            lrates = (W @ source.rates) * a * type
            where:
              - W is row-normalized
              - a is per-target positive amplitude (homeostatic gain)
              - type is +1 (E) or -1 (I)
        '''
        self.lrates[:] = (self.W @ self.source.rates) * self.a *self.type

    """
    Update exponential sliding STFT-like stats for each frequency band.
    For each band k:
        band[k]["freq"]: cycles per step (1/period)
        band[k]["tau"]:  exp smoothing factor in (0,1)

    rate_band[k] holds:
        "in"/"out"  : smoothed rates (pre/post)
        "cin"/"sin" : smoothed rate * cos/sin phase (pre)
        "cout"/"sout": same for post

        Used in update_weights() to compute phase-coherence-based learning factors.
        """
    def band_update(self):
        # for each band we need to keep track of the complex STFT amplitude at the target frequency as well as the sliding average of the firing rate.
        # We require tracking cosine, sine and average for the input and output firing rates
        if(len(self.rate_band)>0):
            rj = self.W@(self.source.rates)
            self.rj2 = self.W@(self.source.rates*self.source.rates)
        for k,v in self.rate_band.items():
            self.rate_band[k]["mu"] = smoothing(self.rate_band[k]["mu"],self.rj,self.band[k]["tau"])
        # update the general averages, while we're at it
        self.rate_average[:] = (1-self.tau)*self.rate_average+self.tau*self.target.rates
        self.rate_square[:] = (1-self.tau)*self.rate_square+self.tau*self.target.rates*self.target.rates
        self.rate_in[:] = (1-np.abs(self.tauin))*self.rate_in+np.abs(self.tauin)*self.source.rates
        self.rate_out[:] = (1-np.abs(self.tauout))*self.rate_out+np.abs(self.tauout)*self.target.rates

    # correlation and CV calculations
    def mixing_update(self):
        self.rit[:] = smoothing(self.rit,self.target.rates,self.tauf)
        self.rjt[:] = smoothing(self.rjt,self.source.rates,self.tauf)
        self.ri2t[:] = smoothing(self.ri2t,self.target.rates*self.target.rates,self.tauf)
        self.rj2t[:] = smoothing(self.rj2t,self.source.rates*self.source.rates,self.tauf)
        self.rs[:] = smoothing(self.rs,self.W@self.source.rates,self.tauf)
        self.r2s[:] = W@(self.source.rates*self.source.rates)
        self.H[:] = smoothing(self.H,(self.W@(self.source.rates-self.rjt))*(self.target.rates-self.rit),self.tauf)
        self.sigi[:] = torch.sqrt(torch.clamp(self.ri2t-self.rit*self.rit,min=0))
        self.sigj[:] = torch.sqrt(torch.clamp(self.rj2t-self.rjt*self.rjt,min=0))
        self.C_fast[:] = self.H/(self.sigi*(self.W@self.sigj)+1e-8)
        self.C[:] = smoothing(self.C,self.C_fast,self.taus)
        self.C2[:] = smoothing(self.C2,torch.abs(self.C_fast),self.taus)
        self.CVt_fast[:] = torch.sqrt(torch.clamp(self.ri2t/(self.rit*self.rit+1e-8)-1,min=0))
        self.CVt[:] = smoothing(self.CVt,self.CVt_fast,self.taus)
        self.CVs_fast[:] = torch.sqrt(torch.clamp((self.r2s+1e-9)/(self.rs*self.rs+1e-8)-1,min=0))
        self.CVs[:] = smoothing(self.CVs,self.CVs_fast,self.taus)

    def pv_decorrelation(self):
        t = self.pooling["target"]
        fb = self.pooling["fast"]
        mb = self.pooling["middle"]
        sb = self.pooling["slow"]

        # watch out, we're applying python trickery here. The variable t will become one of two different types of class object. We assume rate_band and band exist as members in class population and in class compartment
        if(t not in self.target.compartments):
            t = self.target
        else:
            t = self.target.compartments["target"]

        ts = t.band[sb]["tau"]
        # get the long term variation of the fast, mid and slow bands
        LP = R-t.rate_band[fb]
        self.Pf = smoothing(self.Pf,LP*LP,ts)
        LP = t.rate_band[fb]-t.rate_band[mb]
        self.Pm = smoothing(self.Pm,LP*LP,ts)
        LP = t.rate_band[mb]-t.rate_band[sb]
        self.Ps = smoothing(self.Ps,LP*LP,ts)
        Ptot = self.Pf+self.Pm+self.Ps

        # band frequencies where fast band power is too low or slow band power is too high. Should have same effect on sign of inhibition.
        Regf = torch.logical_or(self.Pf/(Ptot+1e-8)<self.theta_f)
        Regs = torch.logical_or(self.Ps/(Ptot+1e-8)>self.theta_s)

        self.mixing[:] = Regf*self.eta_f + Reg_s*self.eta_s

        # make sure direction of inhibition is correct for various cases
        # too low CV region. Inhibition should go up
        # assign explicitly to avoid numerical drift... but that shouldn't matter
        
        '''
        self.dir[:] = self.dir*torch.logical_not(RegA) + RegA
        #self.dir+=RegA*(1-self.dir)
        # minimum wait to potentially change direction of inhibition
        
        r = torch.randn_like(self.waiting)
        noskip = torch.logical_and(self.waiting>self.skip*3/4,r<4/self.skip)

        # not inside region A, and if correlation increased, we should flip the direction
        RegB = torch.logical_and(torch.logical_and(self.C2>self.C_fast,torch.logical_not(RegA)),noskip)
        self.dir[:] = -self.dir*RegB + self.dir*torch.logical_not(RegB)
        #self.dir+=-2*self.dir*RegB
        # update tracking variables for next K-step comparison
        # reset waiting
        self.waiting = self.waiting*torch.logical_not(noskip)
        # pv neurons don't use C_fast so we use it as a placeholder
        self.C_fast[:] = self.C_fast*torch.logical_not(noskip) + self.C2*noskip
        self.waiting+=1
        self.mixing[:]=self.dir*RegA*self.etaA + self.dir*torch.logical_not(RegA)*self.etaB-self.etaC
        '''
        #self.mixing[:] = RegA*self.etaA - torch.logical_not(RegA)*self.etaB



    def normalize_weights(self):
        self.w/=((self.W@self.ones)[self.w_ind[0]]+1e-12)

    def rate_stdp(self,k):
        freq = self.band[k]["freq"]
        c_s = np.cos(freq*2*np.pi)
        s_s = np.sin(freq*2*np.pi)
        c_d = self.rate_band[k]["cin"][self.w_ind[1]]*self.rate_band[k]["cout"][self.w_ind[0]]+self.rate_band[k]["sin"][self.w_ind[1]]*self.rate_band[k]["sout"][self.w_ind[0]]
        s_d = self.rate_band[k]["sin"][self.w_ind[1]]*self.rate_band[k]["cout"][self.w_ind[0]]-self.rate_band[k]["cin"][self.w_ind[1]]*self.rate_band[k]["sout"][self.w_ind[0]]
        ampin = torch.sqrt(self.rate_band[k]["cin"]*self.rate_band[k]["cin"]+self.rate_band[k]["sin"]*self.rate_band[k]["sin"])
        ampout = torch.sqrt(self.rate_band[k]["cout"]*self.rate_band[k]["cout"]+self.rate_band[k]["sout"]*self.rate_band[k]["sout"])
        return self.band[k]["alpha"]*torch.sign(s_d*c_s-c_d*s_s)*((4*(c_d*c_s+s_d*s_s)+ampin[self.w_ind[1]]*ampout[self.w_ind[0]])/(self.rate_band[k]["in"][self.w_ind[1]]*self.rate_band[k]["out"][self.w_ind[0]]+1e-8))

    def update_weights(self):
        """
        Update synaptic weights and compartment amplitude:

        w_ij <- w_ij
            + eta * M * freq_factor_ij
            * (r_pre_j - rin) * (r_post_i - rout)
            - beta * (w_ij - 1/k)

        - freq_factor_ij from band_update(): boosts/suppresses plasticity
            based on oscillatory phase alignment of pre/post.
        - beta term pulls weights toward uniform within each row.
        - After update: clamp >=0 and renormalize rows.

        Amplitude a (per target) is updated in log-space:
           loga += delta * (rate_target - r_post_avg) - rho * (loga - lA0)
        """
        self.band_update()
        temp = torch.zeros(self.nsyn).to(self.net.device)

        # weight learning factor based on frequency band alignment
        # aka, enhance learning for oscillating input and output at target frequencies
        for k in self.band:
            #temp+=self.rate_stdp(k)*torch.clamp((self.source.rates[self.w_ind[1,:]]-self.rate_band[k]["in"][self.w_ind[1,:]])*(self.target.rates[self.w_ind[0,:]]-self.rate_band[k]["out"][self.w_ind[0,:]]),min=0)
            temp+=self.rate_stdp(k)*self.source.rates[self.w_ind[1,:]]*self.target.rates[self.w_ind[0,:]]
        # weight update using reward gating, frequency gating, Hebbian learning and weight narrowing
        if(self.tauin>0):
            hebb = (self.source.rates[self.w_ind[1,:]]-self.rate_in[self.w_ind[1,:]]*self.rin)
        else:
            hebb = (self.source.rates[self.w_ind[1,:]]-self.rin)
        if(self.tauout>0):
            hebb*=(self.target.rates[self.w_ind[0,:]]-self.rate_out[self.w_ind[0,:]]*self.rout)
        else:
            hebb*=(self.target.rates[self.w_ind[0,:]]-self.rout)
        self.w+=temp+self.eta*hebb+self.nu*self.M*self.source.rates[self.w_ind[1,:]]*self.target.rates[self.w_ind[0,:]]-self.beta*(self.w-1/self.k)
        # clamp weights
        self.w.clamp_(min=0)
        self.normalize_weights()
        self.W._values()[:] = self.w
        # adjust compartment amplitudes
        self.loga+=self.delta*torch.log((self.rate_target+1e-3)/(self.rate_average+1e-3))+self.mixing-self.rho*(self.loga-self.lA0)
        self.a[:] = torch.exp(self.loga)

def smoothing(self,tracker,input,tau):
    return (1-tau)*tracker+tau*input

'''
population parameters:
size: width, height and thickness of this neuron population
tau: smoothing time constant of this neuron populations firing rate
activation_exponent: exponent of the joint compartment input (0.5-1 for excitatory and 1-2 for inhibitory populations)
baseline: minimal firing rate of the neuron (after smoothing)
cap: maximal firing rate of the neuron (after smoothing)
activation: function handle. The function should combine the local rates of each compartment in the population (by compartment id). The default in the sum of all populations. Consider using shunting populations
(eg something like:
def shunting(u):
    s = (u["excitation"]+u["inhibition"])/u["shunt_inh"]
    return s
)
'''

def population_parameters(id,size=[28,28,1],tau=0,rate_inflection = 50,activation_exponent=1,baseline=0,cap=300,activation=None,type_ratio=3):
    parameters = {}
    parameters["id"] = id
    parameters["size"] = cp.copy(size)
    parameters["tau"] = 1./(1+tau)
    parameters["p"] = activation_exponent
    parameters["r0"] = rate_inflection
    parameters["ratio"] = type_ratio
    if (baseline<0):
        parameters["baseline"] = 0
    else:
        parameters["baseline"] = baseline
    if(cap<parameters["baseline"]):
        print("Warning: Your firing upper limit is lower than you lower limit.")
    parameters["cap"] = cap
    if(activation==None):
        def simple(u):
            s = u[next(iter(u))].clone()
            s[:] = 0
            for i in u.values():
                s+=i
            return s

        parameters["activation"] = simple
    else:
        parameters["activation"]=activation
    return parameters

'''
compartment parameters:
ellipse: horizontal and vertical axis of the elliptical receptive field of the post synaptic neurons in population grid coordinates
tsyn: intended number of synapses in this compartment per post synaptic target. A negative value means that the receptive field of the neuron covers the whole population of the pre-synaptic population. tsyn might be reduced if the given value exceeds the number of possible source neurons in the receptive field
A: initial amplitude of the summed weight of synapses per neuron in this compartment. Negative values imply that synapses are inhibitory.
A0: target amplitude that the amplitude relaxes towards. Value must be larger than 0 (amplitude learning is multiplicative)
eta: "Hebbian" learning rate of synapse weights
beta: relaxation rate of synapse weights towards equal weights per neuron
band: frequency bands for oscillation learning and phase coincidence. Bands are stored in a dictionary. Each band is {"tau":..., "period":..., "alpha":..., }, where period is an oscillation cycle in steps, tau the sliding window smoothing duration and alpha the strength of frequency learning rate
tau: exponetial sliding average time (long) for the target firing rate of the post synaptic neurons
rin,rout: firing rates flipping Hebbian learning for (mainly inhibitory) synapses
delta: amplitude learning rate
rate_target: target long term firing average for the post synaptic neuron
'''
def compartment_parameters(id,source,target,ellipse=[1,1],tsyn=1,A=2,A0=-1,eta=0,nu=0,beta=0,bands=None,rho=0,tau=0,rin=0,rout=0,tauin=-1,tauout=-1,delta=0,rate_target=0,eps=5,stype="",pooling=None):
    parameters = {}
    parameters["id"] = id
    parameters["source"] = source
    parameters["target"] = target
    parameters["ellipse"] = cp.copy(ellipse)
    parameters["tsyn"] = tsyn
    parameters["A"] = A
    parameters["stype"] = stype
    if (stype=="E"):
        parameters["tauf"] = 1./(1+pooling["tauf"])
        parameters["taus"] = 1./(1+pooling["taus"])
    elif (stype=="PV"):
        parameters["etaA"] = pooling["etaA"]
        parameters["etaB"] = pooling["etaB"]
        parameters["etaC"] = pooling["etaC"]
        parameters["skip"] = pooling["skip"]
        parameters["theta_t"] = pooling["theta_t"]
        parameters["theta_s"] = pooling["theta_s"]
        
    parameters["eps"] = eps
    if(A0>0):
        parameters["A0"] = A0
    else:
        parameters["A0"] = np.sign(A)*A0
    parameters["eta"] = eta/np.abs(tsyn)
    parameters["nu"] = nu/np.abs(tsyn)
    parameters["beta"] = beta
    parameters["delta"] = delta
    parameters["rho"] = rho
    parameters["tau"] = 1./(1+tau)
    parameters["rin"] = rin
    parameters["rout"] = rout
    if(tauin>0):
        parameters["tau"] = 1./(1+tauin)
    else:
        parameters["tau"] = 1./(tauin-1)
    parameters["rate_target"] = rate_target
    parameters["band"] = cp.deepcopy(bands) if bands is not None else {}
    for k in parameters["band"]:
        parameters["band"][k]["freq"] = 1./parameters["band"][k]["period"]
        del parameters["band"][k]["period"]
        parameters["band"][k]["tau"] = 1./(1+parameters["band"][k]["tau"])
        parameters["band"][k]["alpha"] = parameters["band"][k]["alpha"]/np.abs(tsyn)
    return parameters

# functions for receptive field sampling in the network module setup
def sample_from_cylindroid(a, b, z, n, num_vectors):
	vectors = []
	for _ in range(num_vectors):
		vector = torch.zeros(0,3,dtype=torch.int)
		while len(vector) < n:
			# Generate random points within the bounding box
			x_coords = torch.randint(-a, a + 1, (n * 3,))  # Oversample to increase chances
			y_coords = torch.randint(-b, b + 1, (n * 3,))
			z_coords = torch.randint(0, z, (n * 3,))

			# Check which points fall inside the ellipsoid
			mask = (x_coords**2 / a**2 + y_coords**2 / b**2) <= 1
			valid_points = torch.stack((x_coords[mask], y_coords[mask], z_coords[mask]), dim=1)

			valid_points = torch.cat((vector,valid_points),dim=0)
			# Remove duplicates
			unique_points = torch.unique(valid_points, dim=0)

			# Append unique points to the vector until it has n points
			if unique_points.size(0) >= n:
				perm_indices = torch.randperm(unique_points.size(0))
				unique_points = unique_points[perm_indices]
				unique_points = unique_points[:n]  # Take only as many points as needed

			vector = unique_points  # Append new unique points

		vectors.append(vector)
	return torch.stack(vectors)

def sample_from_box(a, b, z, n, num_vectors):
	vectors = []
	for _ in range(num_vectors):
		vector = torch.zeros(0,3,dtype=torch.int)
		while len(vector) < n:
			# Generate random points within the bounding box
			x_coords = torch.randint(0, a, (n * 3,))  # Oversample to increase chances
			y_coords = torch.randint(0, b, (n * 3,))
			z_coords = torch.randint(0, z, (n * 3,))

			valid_points = torch.stack((x_coords, y_coords, z_coords), dim=1)

			valid_points = torch.cat((vector,valid_points),dim=0)
			# Remove duplicates
			unique_points = torch.unique(valid_points, dim=0)

			# Append unique points to the vector until it has n points
			if unique_points.size(0) >= n:
				perm_indices = torch.randperm(unique_points.size(0))
				unique_points = unique_points[perm_indices]
				unique_points = unique_points[:n]  # Take only as many points as needed

			vector = unique_points  # Append new unique points

		vectors.append(vector)
	return torch.stack(vectors)

def integer_points_cylindroid(a, b, z):
	# Generate all integer points within the bounding box
	x = torch.arange(-a, a + 1, dtype=torch.int)
	y = torch.arange(-b, b + 1, dtype=torch.int)
	z = torch.arange(0, z, dtype=torch.int)
	# Create a grid of points
	X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

	# Flatten the grid to have a list of points
	points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

	# Apply the ellipsoid equation to filter points
	if(a==0):
		a=1
	if(b==0):
		b=1
	mask = (points[:, 0].float() / a) ** 2 + (points[:, 1].float() / b) ** 2 <= 1
	ellipsoid_points = points[mask]
	return ellipsoid_points

def get_permutations(points, n, t):
	num_points = points.size(0)
	if t > num_points:
		raise ValueError("t cannot be greater than the number of points.")

	# Initialize a tensor to hold the permutations
	permutations = torch.empty((n, t, points.size(1)), dtype=points.dtype)

	for i in range(n):
		# Generate a random permutation of indices
		perm_indices = torch.randperm(num_points)[:t]
		# Index into the points using the permuted indices and add to the permutations tensor
		permutations[i] = points[perm_indices]
	return permutations

"""
Sample synaptic source indices for each of n target neurons.

- os: source population size (WxHxD)
- ts: target population size (WxHxD)
- tsyn > 0: sample within local ellipsoid (a,b) around each target
- tsyn < 0: sample from entire source population (|tsyn| per target)

Returns:
    points: 1D tensor of length n * k with flattened source indices
    k     : synapses per target (may be <= |tsyn| if limited by geometry)
"""
def sample_synapses(os,ts,a,b,n,tsyn):
	w = os[0]
	h = os[1]
	z = os[2]
	# if the value of tn.syn is negative we sample -tn.syn coordinates from the whole input region
	t = tsyn
	if(tsyn<0):
		t = -t
		if(t<w*h*z*0.25):
			# case where there are few enough synapses to draw without resampling using duplicate rejection
			points = sample_from_box(w, h, z, t, n)
		else:
			x = torch.arange(0, w, dtype=torch.int)
			y = torch.arange(0, h, dtype=torch.int)
			z = torch.arange(0, z, dtype=torch.int)
			X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
			# Flatten the grid to have a list of points
			points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
			if(points.size(0)<t):
				t = points.size(0)
			points = get_permutations(points,n,t)
	else:
			# receptive field size. This is based on the size of the projecting region
		ellipse_a = int(a)
		ellipse_b = int(b)
		# check if the number of synapses for each target is a substantial fraction of the ellipse bounding box (12.5% choosen here) and if a permutation draw needs to be done or not
		if(t<0.5*(ellipse_a*ellipse_b)*z):
			# case where there are few enough synapses to draw without resampling using duplicate rejection
			points = sample_from_cylindroid(ellipse_a, ellipse_b, z, t, n)
		else:
			points = integer_points_cylindroid(ellipse_a, ellipse_b, z)
			if(points.size(0)<t):
				t = points.size(0)
			points = get_permutations(points,n,t)

		# for the ellipsoid receptive field we need to shift the samples according to the target coordinates (after rescaling) and perform a modulo operation for periodic boundary conditions
		scalar_indices = torch.arange(0, n, dtype=torch.int)
		#offset_indices = torch.stack(torch.unravel_index(scalar_indices, ts), dim=1)
		offset_indices = torch.from_numpy(np.column_stack(np.unravel_index(scalar_indices.numpy(), ts)))

		# rescale the indices so the
		offset_indices[:,2] = 0
		offset_indices = (offset_indices*(torch.tensor(os)/torch.tensor(ts))).int()
		points = offset_indices.unsqueeze(1)+points
		points = torch.remainder(points,torch.tensor(os))
	# unravel to single dimensional indices for neurons (will need to be converted back for plotting neurons in the module shape)
	p = points[..., 0] * (os[1] * os[2]) + points[..., 1] * os[2] + points[..., 2]
	p_blocks = p.view(-1, t)

	# Sort each block. torch.sort returns both sorted values and indices, we take the values
	sorted_blocks = torch.sort(p_blocks, dim=1)[0]

	# If you need the sorted vector back in the original flat format:
	points = sorted_blocks.view(-1)

	return points,t

def efficient_append_to_hdf5(file_name, dataset_name, tensor):
    # Convert the PyTorch tensor to a NumPy array
    data = tensor.cpu().numpy()

    with h5py.File(file_name, 'a') as f:  # Open the file in append mode
        if dataset_name in f:
            # The dataset exists, so we'll append the new data
            dset = f[dataset_name]
            # Calculate the new size after appending the new data
            new_size = dset.shape[0] + 1  # Assuming we're appending along the first dimension
            # Resize the dataset to accommodate the new data
            dset.resize(new_size, axis=0)
            # Write the new data to the extended part of the dataset
            dset[-1, :] = data
        else:
            # The dataset does not exist, so create it with unlimited size along the first dimension
            dset = f.create_dataset(dataset_name, data=data[np.newaxis, :], maxshape=(None, data.shape[0]))
