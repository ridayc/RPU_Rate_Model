import torch
import math
import numpy as np
from scipy.special import erfcinv
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
        self.u_eff = torch.zeros(self.nneu).to(self.net.device)
        self.E_eff = torch.zeros(self.nneu).to(self.net.device)
        self.I_eff = torch.zeros(self.nneu).to(self.net.device)
        self.ravg = self.r0
        self.rsq = self.r0*self.r0

    def add_compartment(self,compartment):
        self.compartments[compartment.id] = compartment

    def activations(self):
        u = [{},{}]
        for v in self.compartments.values():
            v.local_rate()
            u[0][v.id] = v.lrates
            u[1][v.id] = v.type
        if len(u[0])>0:
            self.u_eff[:],self.E_eff[:],self.I_eff[:] = self.activation(u)
            self.u_eff.clamp_(min=0)
            self.uact[:] = torch.clamp((self.u_eff**self.p)*self.r0**(1-self.p),min=self.baseline,max=self.cap)
        else:
            self.uact[:] = 0.
            self.u_eff[:] = 1.
            self.I_eff[:] = -3.
            self.E_eff[:] = 1.


    def update_rates(self):
        self.rates[:] = (1-self.tau)*self.rates+self.tau*self.uact
        self.ravg = torch.mean(self.rates).item()
        self.rsq = torch.mean(self.rates*self.rates).item()

    def update_weights(self):
        for v in self.compartments.values():
            v.update_weights()

    def E_I_balance(self):
        counter = 0
        # get
        for v in self.compartments.values():
            if (v.stat):
                v.mixing_update()
            v.band_update()


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
        self.etal = compartment_param["etal"]
        self.etar = compartment_param["etar"]
        self.thetar = compartment_param["thetar"]
        self.alpha = compartment_param["alpha"]
        self.nu = compartment_param["nu"]
        self.beta = compartment_param["beta"]
        self.beta0 = np.log(compartment_param["beta0"])
        self.bn = compartment_param["bn"]
        self.bp = compartment_param["bp"]
        self.kappa = compartment_param["kappa"]
        self.delta = compartment_param["delta"]
        self.rho = compartment_param["rho"]
        #self.gamma = compartment_param["gamma"]
        self.zeta = compartment_param["zeta"]
        self.zeta2 = compartment_param["zeta2"]
        self.tau = compartment_param["tau"]
        self.tauw = compartment_param["tauw"]
        self.taub = compartment_param["taub"]
        self.taul = compartment_param["taul"]
        self.taug = compartment_param["taug"]
        #self.tauz = compartment_param["tauz"]
        self.rin = compartment_param["rin"]
        self.rout = compartment_param["rout"]
        self.tauin = compartment_param["tauin"]
        self.tauout = compartment_param["tauout"]
        #self.cv = compartment_param["cv"]
        # rate quatile for thresholding high rate events for PV neurons (I-I amplitude controller)
        self.rq = compartment_param["rq"]
        self.rt = compartment_param["rt"]
        self.noise = compartment_param["noise"]
        self.cv = compartment_param["cv"]
        self.eps = compartment_param["eps"]
        self.stype = compartment_param["stype"]
        self.M = 0
        self.rate_target = compartment_param["rate_target"]
        self.z_value = compartment_param["z_value"]
        self.thetaz = compartment_param["thetaz"]
        self.ratio = compartment_param["ratio"]
        self.c_c = compartment_param["c_c"]

        if(self.c_c[0]==""):
            self.c_c[0] = self.id
        self.stat = compartment_param["stat"]
        # measurment variables for excitatory compartments. Local correlation and coefficient of variance markers
        if(self.stat):
            self.tauf = compartment_param["tauf"]
            self.taus = compartment_param["taus"]
            self.rit = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
            self.rit_slow = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
            self.rjt = torch.full((self.source.nneu,),self.rate_target).to(self.net.device)
            self.ri2t = torch.full((self.target.nneu,),self.rate_target*self.rate_target).to(self.net.device)
            self.rj2t = torch.full((self.source.nneu,),self.rate_target*self.rate_target).to(self.net.device)
            self.rs = torch.zeros(self.target.nneu).to(self.net.device)
            self.r2s = torch.zeros(self.source.nneu).to(self.net.device)
            self.sigi = torch.zeros(self.target.nneu).to(self.net.device)
            self.sigj = torch.zeros(self.source.nneu).to(self.net.device)
            # local covariance tracker
            self.H = torch.zeros(self.target.nneu).to(self.net.device)

        # ltp and ltd factors
        a0 =  np.sqrt(compartment_param["an"]*compartment_param["ap"])
        # we want the geometric mean of an and ap to be 1
        self.an = torch.full((self.target.nneu,),compartment_param["an"]/a0).to(self.net.device)
        self.ap = torch.full((self.target.nneu,),compartment_param["ap"]/a0).to(self.net.device)
        # expected weight gain over time
        self.E_dw = torch.zeros(self.target.nneu).to(self.net.device)
        self.dM = torch.full((self.target.nneu,),np.log(compartment_param["ap"]/compartment_param["an"])).to(self.net.device)
        self.dN = torch.zeros(self.target.nneu).to(self.net.device)
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
        
        self.rate_q = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.rate_average = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        self.rate_square = torch.full((self.target.nneu,),2*self.rate_target*self.rate_target).to(self.net.device)
        self.CV_slow = torch.full((self.target.nneu,),self.cv).to(self.net.device)
        self.fast_average = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        '''
        self.E_smooth = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.I_smooth = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.F_smooth = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.U_smooth = torch.full((self.target.nneu,),0.).to(self.net.device)
        '''
        self.numerator = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.denominator = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.mu_E = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.mu_I = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.mu2_E = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.mu2_I = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.rate_in = torch.full((self.source.nneu,),self.rate_target).to(self.net.device)
        self.rate_out = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        self.band = cp.deepcopy(compartment_param["band"])
        self.rate_band = {}
        if("amplitude" in self.band):
            self.rate_band["amplitude"] = {}
            amp = self.rate_band["amplitude"]
            amp["target"] = self.band["amplitude"]["target"]
            amp["tau"] = cp.deepcopy(self.band["amplitude"]["tau"])
            amp["taup"] = cp.deepcopy(self.band["amplitude"]["taup"])
            amp["theta"] = cp.deepcopy(self.band["amplitude"]["theta"])
            amp["eta"] = cp.deepcopy(self.band["amplitude"]["eta"])
            # go through the band power variables
            for i in ["r","mu","s2","p"]:
                amp[i] = {}
                # go through the bands of each band variable
                for j in ["f","m","s"]:
                    amp[i][j] = torch.zeros(self.target.nneu).to(self.net.device)
        if("synapse" in self.band):
            self.rate_band["synapse"] = {}
            sizes = {"in":self.source.nneu,"out":self.target.nneu}
            for k in ["in","out"]:
                syn = self.rate_band["synapse"]
                syn[k] = {}
                syn[k]["tau"] = cp.deepcopy(self.band["synapse"][k]["tau"])
                syn[k]["taup"] = cp.deepcopy(self.band["synapse"][k]["taup"])
                syn[k]["theta"] = cp.deepcopy(self.band["synapse"][k]["theta"])
                syn[k]["eta"] = cp.deepcopy(self.band["synapse"][k]["eta"])
                # go through the band power variables
                for i in ["r","mu","s2","p"]:
                    syn[k][i] = {}
                    # go through the bands of each band variable
                    for j in ["f","m","s"]:
                        syn[k][i][j] = torch.zeros(sizes[k]).to(self.net.device)
        self.freq = cp.deepcopy(compartment_param["freq"])
        self.freq_band = {}
        for k in self.freq:
            self.freq_band[k] = {"in": torch.full((self.source.nneu,),0,device=self.net.device), "out": torch.full((self.target.nneu,),0,device=self.net.device),"cin": torch.full((self.source.nneu,),0,device=self.net.device), "sin": torch.full((self.source.nneu,),0,device=self.net.device),"cout": torch.full((self.target.nneu,),0,device=self.net.device), "sout": torch.full((self.target.nneu,),0,device=self.net.device)}

        target_size = torch.Size(self.target.size)
        origin_size = torch.Size(self.source.size)
        points,self.k = sample_synapses(origin_size,target_size,compartment_param["ellipse"][0],compartment_param["ellipse"][0],math.prod(self.target.size),compartment_param["tsyn"])
        self.ones = torch.full((self.source.nneu,),1.).to(self.net.device)
        self.nsyn = self.target.nneu*self.k
        self.inds = torch.arange(self.target.nneu).repeat_interleave(self.k).to(self.net.device)
        self.indt = points.view(-1).to(self.net.device)
        self.w_ind = torch.stack((self.inds,self.indt))
        self.w = torch.zeros(self.nsyn).to(self.net.device)
        self.band_gains = torch.full((self.nsyn,),0.).to(self.net.device)
        self.temp_freq = torch.full((self.nsyn,),0.).to(self.net.device)
        self.hebb = torch.zeros(self.nsyn).to(self.net.device)
        self.dw = torch.zeros(self.nsyn).to(self.net.device)
        #self.w+= 1./self.k
        self.w[:] = torch.exp(self.eps*torch.randn_like(self.w))
        self.eta/=self.k
        self.alpha/=self.k
        self.nu/=self.k
        self.thetar/=self.k
        for k in self.freq:
            self.freq[k]["alpha"] = self.freq[k]["alpha"]/self.k
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
        self.SOM = compartment_param["SOM"]
        if(self.SOM!=None):
            self.SOM.setup(self)


    def local_rate(self):
        '''
        Compute this compartment's contribution to the target population:
            lrates = (W @ source.rates) * a * type
            where:
              - W is row-normalized
              - a is per-target positive amplitude (homeostatic gain)
              - type is +1 (E) or -1 (I)
        '''
        if(self.SOM!=None and self.SOM.type=="post"):
            self.lrates[:] = self.SOM.activation()
        else:
            self.lrates[:] = (self.W @ self.source.rates) * self.a * self.type



    def band_update(self):
        # 
        if("amplitude" in self.rate_band):
            # frequencies should be based on firing input in a particular compartment
            amp =  self.rate_band["amplitude"]
            if(amp["target"] in self.target.compartments):
                c = self.target.compartments[amp["target"]]
                self.rate_frequencies(c.lrates,amp["r"],amp["tau"])
                self.band_power(c.lrates,amp,amp["taup"])
            # otherwise frequencies are based on the neurons output rate
            else:
                self.rate_frequencies(self.target.rates,amp["r"],amp["tau"])
                self.band_power(self.target.rates,amp,amp["taup"])
        
        if("synapse" in self.rate_band):
            syn = self.rate_band["synapse"]
            # post side
            self.rate_frequencies(self.target.rates,syn["out"]["r"],syn["out"]["tau"])
            self.band_power(self.target.rates,syn["out"],syn["out"]["taup"])
            # pre side
            self.rate_frequencies(self.source.rates,syn["in"]["r"],syn["in"]["tau"])
            self.band_power(self.source.rates,syn["in"],syn["in"]["taup"])

        for k,v in self.freq_band.items():
            freq = self.freq[k]["freq"]
            tau = self.freq[k]["tau"]
            cs = np.cos((self.net.time*freq)%1*2*np.pi)
            sn = np.sin((self.net.time*freq)%1*2*np.pi)
            v["in"][:] = (1-tau)*v["in"]+tau*self.source.rates
            v["out"][:] = (1-tau)*v["out"]+(tau)*self.target.rates
            v["cin"][:] = (1-tau)*v["cin"]+(tau*cs)*self.source.rates
            v["sin"][:] = (1-tau)*v["sin"]+(tau*sn)*self.source.rates
            v["cout"][:] = (1-tau)*v["cout"]+(tau*cs)*self.target.rates
            v["sout"][:] = (1-tau)*v["sout"]+(tau*sn)*self.target.rates
        
        # update the general averages, while we're at it
        #smoothing(self.rate_square,self.target.rates*self.target.rates,self.taub)
        if(self.rq>0):
            smoothing(self.rate_average,self.target.rates,self.tau)
            smoothing(self.rate_square,self.target.rates*self.target.rates,self.taub)
            smoothing(self.rate_q,(self.target.rates>self.rt*self.rate_average).float(),self.tau)
        else:
            smoothing(self.rate_average,self.target.rates,self.tau)
            smoothing(self.rate_square,self.target.rates*self.target.rates,self.taub)
            if(self.cv>0):
                smoothing(self.fast_average,self.target.rates,self.taub)
                smoothing(self.CV_slow,self.fast_average*torch.clamp(self.rate_square/(self.fast_average*self.fast_average+1e-8)-1,min=0),self.tau)
        smoothing(self.rate_in,self.source.rates,np.abs(self.tauin))
        smoothing(self.rate_out,self.target.rates,np.abs(self.tauout))

    # get the smoothing of the rates according to the target smoothing length (either multiple rates one tau, or one rate and multiple taus). This overwrites the values of the tensor(s) in rates
    def rate_frequencies(self,rates,rate_band,tau):
        for i in rate_band:
            if(isinstance(tau,dict)):
                smoothing(rate_band[i],rates,tau[i])
            else:
                smoothing(rate_band[i],rates[i],tau)

    def band_power(self,rates,power_band,tau):
        # get the specific frequency bands for fast, mid and slow
        LP = {"f":rates-power_band["r"]["f"],"m":power_band["r"]["f"]-power_band["r"]["m"],"s":power_band["r"]["m"]-power_band["r"]["s"]}
        # calculate the long term band averages
        self.rate_frequencies(LP,power_band["mu"],tau)
        # calculate the long term band squared averages
        for i in LP:
            LP[i][:] = LP[i]*LP[i]
        self.rate_frequencies(LP,power_band["s2"],tau)
        for i in power_band["p"]:
            power_band["p"][i][:] = smooth_variance(power_band["s2"][i],power_band["mu"][i])

    def rate_stdp(self,k):
        freq = self.freq[k]["freq"]
        c_s = np.cos(freq*2*np.pi)
        s_s = np.sin(freq*2*np.pi)
        c_d = self.freq_band[k]["cin"][self.w_ind[1]]*self.freq_band[k]["cout"][self.w_ind[0]]+self.freq_band[k]["sin"][self.w_ind[1]]*self.freq_band[k]["sout"][self.w_ind[0]]
        s_d = self.freq_band[k]["sin"][self.w_ind[1]]*self.freq_band[k]["cout"][self.w_ind[0]]-self.freq_band[k]["cin"][self.w_ind[1]]*self.freq_band[k]["sout"][self.w_ind[0]]
        ampin = torch.sqrt(self.freq_band[k]["cin"]*self.freq_band[k]["cin"]+self.freq_band[k]["sin"]*self.freq_band[k]["sin"])
        ampout = torch.sqrt(self.freq_band[k]["cout"]*self.freq_band[k]["cout"]+self.freq_band[k]["sout"]*self.freq_band[k]["sout"])
        return self.freq[k]["alpha"]*torch.sign(s_d*c_s-c_d*s_s)*((4*(c_d*c_s+s_d*s_s)+ampin[self.w_ind[1]]*ampout[self.w_ind[0]])/(self.freq_band[k]["in"][self.w_ind[1]]*self.freq_band[k]["out"][self.w_ind[0]]+1e-8))


    # correlation and CV calculations
    def mixing_update(self):
        smoothing(self.rit,self.target.rates,self.tauf)
        smoothing(self.rit_slow,self.rit,self.taus)
        smoothing(self.rjt,self.source.rates,self.tauf)
        smoothing(self.ri2t,self.target.rates*self.target.rates,self.tauf)
        smoothing(self.rj2t,self.source.rates*self.source.rates,self.tauf)
        smoothing(self.rs,self.lrates,self.tauf)
        self.r2s[:] = (self.W@(self.source.rates*self.source.rates))*self.a*self.a
        smoothing(self.H,(self.W@(self.source.rates-self.rjt))*(self.target.rates-self.rit),self.tauf)
        self.sigi[:] = torch.sqrt(torch.clamp(self.ri2t-self.rit*self.rit,min=0))
        self.sigj[:] = torch.sqrt(torch.clamp(self.rj2t-self.rjt*self.rjt,min=0))
        self.C_fast[:] = self.H/(self.sigi*(self.W@self.sigj)+1e-8)
        smoothing(self.C,self.C_fast,self.taus)
        smoothing(self.C2,torch.abs(self.C_fast),self.taus)
        self.CVt_fast[:] = torch.sqrt(torch.clamp(self.ri2t/(self.rit*self.rit+1e-8)-1,min=0))
        smoothing(self.CVt,self.CVt_fast*self.rit,self.taus)
        self.CVs_fast[:] = torch.sqrt(torch.clamp((self.r2s+1e-9)/(self.rs*self.rs+1e-8)-1,min=0))
        smoothing(self.CVs,self.CVs_fast,self.taus)

    def amplitude_power(self):
        amp = self.rate_band["amplitude"]
        Ptot = amp["p"]["f"]+amp["p"]["m"]+amp["p"]["s"]+1e-8

        # band frequencies dead bands for fast and slow bands
        #Regf1 = amp["p"]["f"]/(Ptot)<amp["theta"]["f"][0]
        #Regf2 = amp["p"]["f"]/(Ptot)>amp["theta"]["f"][1]
        #Regs1 = amp["p"]["s"]/(Ptot)<amp["theta"]["s"][0]
        #Regs2 = amp["p"]["s"]/(Ptot)>amp["theta"]["s"][1]

        #return Regf1*amp["eta"]["f"][0] - Regf2*amp["eta"]["f"][1] - Regs1*amp["eta"]["s"][0] + Regs2*amp["eta"]["s"][1]
        # ramp to threshold approach for the band regions
        return torch.clamp(amp["theta"]["f"][0]-amp["p"]["f"]/(Ptot),min=0)*amp["eta"]["f"][0]+torch.clamp(amp["p"]["f"]/(Ptot)-amp["theta"]["f"][1],min=0)*amp["eta"]["f"][1]-torch.clamp(amp["theta"]["s"][0]-amp["p"]["s"]/(Ptot),min=0)*amp["eta"]["s"][0]+torch.clamp(amp["p"]["s"]/(Ptot)+amp["theta"]["s"][1],min=0)*amp["eta"]["s"][1]

    def synaptic_band_gain(self):
        post = self.rate_band["synapse"]["out"]
        pre = self.rate_band["synapse"]["in"]
        # get total power for all input and output neurons
        ppost = post["p"]["f"]+post["p"]["m"]+post["p"]["s"]+1e-8
        ppre = pre["p"]["f"]+pre["p"]["m"]+pre["p"]["s"]+1e-8

        # post gain regions
        Regf1 = post["p"]["f"]/(ppost)<post["theta"]["f"][0]
        Regf2 = post["p"]["f"]/(ppost)>post["theta"]["f"][1]
        Regs1 = post["p"]["s"]/(ppost)<post["theta"]["s"][0]
        Regs2 = post["p"]["s"]/(ppost)>post["theta"]["s"][1]

        # post gain
        fact = (Regf1*post["eta"]["f"][0]+Regf2*post["eta"]["f"][1]+torch.logical_not(torch.logical_or(Regf1,Regf2)))*(Regs1*post["eta"]["s"][0]+Regs2*post["eta"]["s"][1]+torch.logical_not(torch.logical_or(Regs1,Regs2)))
        sign = (Regf1|Regf2|Regs1|Regs2)
        gpost = -fact*sign+torch.logical_not(sign)

        # pre gain regions
        Regf1 = pre["p"]["f"]/(ppre)<pre["theta"]["f"][0]
        Regf2 = pre["p"]["f"]/(ppre)>pre["theta"]["f"][1]
        Regs1 = pre["p"]["s"]/(ppre)<pre["theta"]["s"][0]
        Regs2 = pre["p"]["s"]/(ppre)>pre["theta"]["s"][1]

        # pre gain
        fact = (Regf1*pre["eta"]["f"][0]+Regf2*pre["eta"]["f"][1]+torch.logical_not(torch.logical_or(Regf1,Regf2)))*(Regs1*pre["eta"]["s"][0]+Regs2*pre["eta"]["s"][1]+torch.logical_not(torch.logical_or(Regs1,Regs2)))
        sign = (Regf1|Regf2|Regs1|Regs2)
        gpre = fact*sign+torch.logical_not(sign)

        # combine post and pre synaptic gain per synapse and return
        return gpost[self.w_ind[0]]*gpre[self.w_ind[1]]

    '''
    def E_I_gain(self):
        # expect all elements in I_eff < 0
        self.E_smooth[:] = smoothing(self.E_smooth,self.target.E_eff,self.taug)
        self.I_smooth[:] = smoothing(self.I_smooth,self.target.I_eff,self.taug)
        return self.E_smooth/(self.E_smooth-self.I_smooth+1e-8)-self.ratio

    def ff_z_score(self):
        self.F_smooth[:] = smoothing(self.F_smooth,self.lrates,self.tauz)
        # hopefully we checked that u_eff is clamped at zero
        self.U_smooth[:] = smoothing(self.U_smooth,self.target.u_eff,self.tauz)
        strong = self.F_smooth>self.thetaz
        score = self.z_value-self.F_smooth/(self.U_smooth+1e-8)
        return strong*score
    '''

    def compartment_gain(self):
        if(self.ratio=="E/I"):
            smoothing(self.numerator,self.target.E_eff,self.taug)
            smoothing(self.denominator,self.target.E_eff-self.target.I_eff,self.taug)
        elif(self.ratio=="tot"):
            smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            smoothing(self.denominator,self.target.E_eff-self.target.I_eff,self.taug)
        elif(self.ratio=="Eeff"):
            smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            smoothing(self.denominator,self.target.E_eff,self.taug)
        elif(self.ratio=="Ieff"):
            smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            smoothing(self.denominator,torch.abs(self.target.I_eff),self.taug)
        elif(self.ratio=="ueff"):
            smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            smoothing(self.denominator,self.target.u_eff+torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
        else:
            smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            smoothing(self.denominator,torch.abs(self.target.compartments[self.c_c[0]].lrates)+torch.abs(self.target.compartments[self.c_c[1]].lrates),self.taug)
        strong = self.numerator>=self.thetaz
        score = self.z_value-self.numerator/(self.denominator+1e-8)
        return score*strong

    def correlation_gain(self):
        smoothing(self.mu_E,self.target.E_eff,self.taug)
        smoothing(self.mu_I,self.target.I_eff,self.taug)
        smoothing(self.mu2_E,self.target.E_eff*self.target.E_eff,self.taug)
        smoothing(self.mu2_I,self.target.I_eff*self.target.I_eff,self.taug)
        # NMC is normalized magnitude coupling. It measures co-fluctuations as opposed to correlations (essentially the absolute coupling terms are taken)
        if(self.ratio=="NMC"):
            smoothing(self.numerator,torch.abs((self.target.E_eff-self.mu_E)*(self.target.I_eff-self.mu_I)/((torch.sqrt(torch.clamp(self.mu2_E-self.mu_E*self.mu_E,min=0))+1e-8)*(torch.sqrt(torch.clamp(self.mu2_I-self.mu_I*self.mu_I,min=0))+1e-8))),self.taug)
            return self.numerator-self.z_value

        else:
            smoothing(self.numerator,(self.target.E_eff-self.mu_E)*(self.target.I_eff-self.mu_I)/((torch.sqrt(torch.clamp(self.mu2_E-self.mu_E*self.mu_E,min=0))+1e-8)*(torch.sqrt(torch.clamp(self.mu2_I-self.mu_I*self.mu_I,min=0))+1e-8)),self.taug)
            return torch.abs(self.numerator)-self.z_value

    def normalize_weights(self):
        self.W._values()[:] = self.w
        self.w/=((self.W@self.ones)[self.w_ind[0,:]]+1e-12)
        self.W._values()[:] = self.w

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
        self.dw[:] = 0
        fr = False
        for i in self.freq_band:
            if(self.freq[i]["alpha"]!=0):
                fr = True
        if(self.alpha!=0 or fr or self.nu!=0):
            self.band_gains[:] = 0
            if("synapse" in self.rate_band and self.alpha>0):
                self.band_gains[:] = self.alpha*self.synaptic_band_gain()

            self.temp_freq[:] = 0
            for i in self.freq_band:
                if(self.freq[i]["alpha"]>0):
                    self.temp_freq+=self.rate_stdp(i)
            self.dw+=(self.band_gains+self.temp_freq+self.nu*self.M)*self.target.rates[self.w_ind[0,:]]*self.source.rates[self.w_ind[1,:]]
        if(self.eta!=0):
            # cov, hebbian and anti-hebbian like learning (a bit different for SOM)
            if(self.SOM!=None and self.SOM.type=="pre"):
                self.hebb[:] = self.SOM.synapse()[self.w_ind[0,:]]
            else:
                if(self.tauout<0):
                     self.hebb[:] = ((self.target.rates-self.rout*self.rate_out)*self.target.rates)[self.w_ind[0,:]]
                else:
                    self.hebb[:] = (self.target.rates-self.rout*self.rate_out)[self.w_ind[0,:]]
            if(self.tauin<0):
                self.hebb*=((self.source.rates-self.rin*self.rate_in)*self.source.rates)[self.w_ind[1,:]]
            else:
                self.hebb*=(self.source.rates-self.rin*self.rate_in)[self.w_ind[1,:]]
                
            self.dw+=self.hebb*self.eta


        

        # ltp and ltd adapation for large synapses
        self.dw[:]=self.an[self.w_ind[0,:]]*torch.clamp(self.dw,max=0)*(torch.pow((1+self.k*self.w)*0.5,self.bn))+self.ap[self.w_ind[0,:]]*torch.clamp(self.dw,min=0)*(torch.pow((1+self.k*self.w)*0.5,-self.bp))

        # structure based noise injection. This should be used in cases where the synapse weight distribution has collapsed to all weights being equal.
        #if(self.noise>0):
        #    self.dw+=2*self.noise*(torch.rand_like(self.w) - 0.5)*self.w  # uniform noise between [-1, 1]

        # L2 regularizer + synapse turnover (because of L1 normalization of weights the form below should have a net zero weight change)
        # that means we use this as a ltp/ltd independent weight distribution regularizer
        if(self.beta>0):
            self.dw+=self.beta*(1/self.k-self.w)*(torch.exp(self.dN))[self.w_ind[0,:]]

        self.dw+=self.w
        # clamp at zero for normalization and total mass drift estimate
        self.dw.clamp_(min=0)

        # prepare for an/ap adaptation
        # smoothed norm calculation
        if(self.etal>0):
            # push an/ap ratio towards balanced ltp/ltd averages over all learning
            smoothing(self.E_dw,row_sum(self.dw-self.w,self.w_ind[0,:]),self.taul)
            self.dM+=-self.etal*self.E_dw
            self.ap[:] = torch.exp(self.dM*0.5)
            self.an[:] = torch.exp(-self.dM*0.5)

        # old idea was based on using synapse sparsity as a comparison measure. We are more explicit here. We say regularization and learning should balance each other out on short timescales (even though the net effect of the regularizer on dw is zero, the important part is the effect it has on weights at the boundary to zero!)
        if(self.etar>0):
            # consider smoothing estimate of Neff but should be needed.
            self.dN+=(1/self.k*row_sum((self.w<self.thetar).float(),self.w_ind[0,:])-self.kappa)*self.etar
            self.dN.clamp_(min=self.beta0)


            '''
            else:
                # abuse of temporary variable hebb as a temporal weight storage
                self.hebb = self.w*self.k-1
                self.dw-=(self.beta/self.k)*torch.sign(self.hebb)*(torch.abs(self.hebb)**self.kappa)
        '''

        # back to smoothing and renomalization
        normalize_by_row(self.dw,self.w_ind[0,:])
        smoothing(self.w,self.dw,self.tauw)
        # clamp weights
        self.w.clamp_(min=0)
        self.normalize_weights()

        # adjust compartment amplitudes
        # balance inhbition and excitation gains
        '''
        if(self.gamma>0):
            self.loga+=self.gamma*self.E_I_gain()
        if(self.zeta>0):
            self.loga+=self.zeta*self.ff_z_score()
        '''
        if(self.zeta!=0):
            if(self.ratio=="corr" or self.ratio=="NMC"):
                self.denominator[:] = self.zeta*self.correlation_gain()
                self.loga+=torch.clamp(self.denominator,min=0)+self.amplitude_power()*(self.denominator<0)
            else:
                self.mu_E[:] = self.compartment_gain()
                self.loga+=self.zeta*torch.clamp(self.mu_E,min=0)+self.zeta2*torch.clamp(self.mu_E,max=0)
            
        #if("amplitude" in self.rate_band):
        #    self.loga+=self.amplitude_power()
        if(self.rho>0):
            self.loga-=self.rho*(self.loga-self.lA0)
            # general leakage term to reduce synapse size; should generally be smaller that any other active (non-balanced) homeostasis terms
            #self.loga-=self.rho
        if(self.delta!=0):
            if(self.rq>0):
                # should be positive for excitation
                #self.loga+=self.delta*(torch.log(np.abs(self.cv)/(CV(self.rate_average,self.rate_square)+1e-8)))
                self.loga+=self.delta*(self.rate_q-self.rq)
            else:
                #self.loga+=self.delta*torch.log((self.rate_target+1e-6)/(self.rate_average+1e-6))
                self.loga+=self.delta*torch.log((self.rate_target+1e-6)/(self.rate_average+1e-6))
                if(self.noise>0):
                    self.loga+=self.delta*self.noise*(self.cv-self.CV_slow/(self.rate_average+1e-8))
                #-0.5*torch.log(1+CV(self.rate_average,self.rate_square))
        self.a[:] = torch.exp(self.loga)

def smoothing(tracker,input,tau):
    tracker.mul_(1 - tau).add_(tau * input)

def smooth_variance(x2,x):
    return torch.clamp(x2-x*x,min=0)

def CV(m,m2):
    return torch.clamp(m2/(m*m+1e-8)-1,min=0)

def normalize_by_row(W, row_idx):
    """
    W:        1D tensor of weights, shape (N,)
    group_idx:1D LongTensor, group assignment for each element, shape (N,)
    """

    # clamp to guarantee weight plausibility
    W.clamp_(min=0.0)

    # compute sum of each group
    row_sums = torch.zeros(row_idx.max()+1, device=W.device)
    row_sums.index_add_(0, row_idx, W)

    # avoid division by zero
    row_sums.clamp_(min=1e-12)

    # normalize each element by its group's sum
    W /= row_sums[row_idx]

def row_sum(value,row_idx):
    row_sums = torch.zeros(row_idx.max()+1, device=value.device)
    return row_sums.index_add_(0, row_idx, value)



# simplest example of a SOM neuron activation function for the output side of synaptic learning
class SOM:
    def __init__(self,som_type,eta_k,omega=2,rho=0.3,c=1,gamma=1):
        self.omega = omega
        self.type = som_type
        self.c = c
        # rho must be between 0 and 1
        self.rho = rho
        self.gamma = gamma
        self.eta_k = eta_k

    def setup(self,comp):
        self.comp = comp
        self.avg = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
        self.sqr = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
        self.g = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
        self.z = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
        self.k = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
        self.quantile = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
        # get the size of the other neurons population to average their Eeff
        if(self.type=="pre"):
            self.avg[:] = 1.
            self.tau = np.abs(self.comp.tauin)
        if(self.type=="post"):
            self.avg[:] = self.comp.source.r0*self.c
            self.sqr = (self.avg+self.comp.source.r0)*(self.avg+self.comp.source.r0)
            self.tau = np.abs(self.comp.tauout)
        self.quantile[:] = self.rho
        

    def zqt_gating(self,z):
        smoothing(self.avg,z,self.tau)
        smoothing(self.sqr,z*z,self.tau)
        self.zq_gating(z,self.avg,torch.sqrt(torch.clamp(self.sqr-self.avg*self.avg,min=0)))

    def zqx_gating(self,z):
        self.zq_gating(z,z.mean(),z.std(unbiased=False))


    def zq_gating(self,z,mu,sigma):
        # z = (x-mu)/sigma
        # g = ([z-k]_+)^gamma
        self.g[:] = torch.clamp((z-mu)/(sigma+1e-8)-self.k,min=0)**self.gamma
        # adapt k to approach longterm rank target
        smoothing(self.quantile,self.g>0,self.tau)
        self.k+=(self.quantile/self.rho-1)*self.eta_k


    def activation(self):
        self.zqt_gating((self.comp.W@(self.comp.source.rates**self.omega))**(1/self.omega))
        return self.g*(self.comp.a*self.comp.type)

    # here burst is the dendritic activity (model with Eeff) of the post-neuron the SOM targets. This is for synaptic learning. Inhibition is handled with the standard lrate approach based on the SOM neuron's firing.
    def synapse(self):
        self.zqx_gating(self.comp.target.E_eff)
        smoothing(self.avg,self.g,self.tau)
        return self.g-self.avg

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

def population_parameters(id,size=[28,28,1],tau=0,rate_inflection = 50,activation_exponent=1,baseline=0,cap=300,activation=None):
    parameters = {}
    parameters["id"] = id
    parameters["size"] = cp.copy(size)
    parameters["tau"] = 1./(1+tau)
    parameters["p"] = activation_exponent
    parameters["r0"] = rate_inflection
    if (baseline<0):
        parameters["baseline"] = 0
    else:
        parameters["baseline"] = baseline
    if(cap<parameters["baseline"]):
        print("Warning: Your firing upper limit is lower than you lower limit.")
    parameters["cap"] = cap
    if(activation==None):
        parameters["activation"] = default_activation
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
def compartment_parameters(id,source,target,ellipse=[1,1],tsyn=1,A=2,A0=-1,eta=0,etal=0,etar=0,alpha=0,nu=0,beta=0,beta0=1e-4,kappa=0,an=1,ap=1,bn=0,bp=0,c_c=None,zeta=0,zeta2="",z_value=0,thetaz=0,ratio="E/I",bands=None,rho=0,tau=0,taug=0,tauw=0,taub=0,taul=0,taur=0,thetar=0,rin=1,rout=1,tauin=1,tauout=1,rq=0,rt=1,noise=0,cv=0,delta=0,rate_target=1,eps=1,stype="",stat=False,power=None,freq=None,SOM=None):
    parameters = {}
    parameters["id"] = id
    parameters["source"] = source
    parameters["target"] = target
    parameters["ellipse"] = cp.copy(ellipse)
    parameters["tsyn"] = tsyn
    parameters["A"] = A
    parameters["stype"] = stype
    parameters["stat"] = stat
    if (stat):
        parameters["tauf"] = 1./(1+power["tauf"])
        parameters["taus"] = 1./(1+power["taus"])
    
    parameters["eps"] = eps
    if(A0>0):
        parameters["A0"] = A0
    else:
        parameters["A0"] = np.sign(A)*A0
    parameters["eta"] = eta
    parameters["etal"] = etal
    parameters["etar"] = etar
    parameters["alpha"] = alpha
    parameters["nu"] = nu
    parameters["beta"] = beta
    parameters["beta0"] = beta0
    parameters["thetar"] = thetar
    parameters["an"] = an
    parameters["ap"] = ap
    parameters["bn"] = bn
    parameters["bp"] = bp
    parameters["kappa"] = kappa
    parameters["zeta"] = zeta
    if(zeta2==""):
        parameters["zeta2"] = zeta
    else:
        parameters["zeta2"] = zeta2
    parameters["c_c"] = cp.copy(c_c) if c_c is not None else ["",""]
    parameters["delta"] = delta
    parameters["rho"] = rho
    parameters["tau"] = 1./(1+tau)
    parameters["taug"] = 1./(1+taug)
    parameters["tauw"] = 1./(1+tauw)
    parameters["taub"] = 1./(1+taub)
    parameters["taul"] = 1./(1+taul)
    parameters["rin"] = rin
    parameters["rout"] = rout
    parameters["rq"] = rq
    parameters["rt"] = rt
    parameters["noise"] = noise
    parameters["cv"] = cv
    parameters["z_value"] = z_value
    parameters["thetaz"] = thetaz
    parameters["ratio"] = ratio
    if(tauin>=0):
        parameters["tauin"] = 1./(1+tauin)
    else:
        parameters["tauin"] = 1./(tauin-1)
    if(tauout>=0):
        parameters["tauout"] = 1./(1+tauout)
    else:
        parameters["tauout"] = 1./(tauout-1)
    parameters["rate_target"] = rate_target
    parameters["band"] = cp.deepcopy(bands) if bands is not None else {}
    if("amplitude" in parameters["band"]):
        parameters["band"]["amplitude"]["taup"] = 1./(1+bands["amplitude"]["taup"])
        for i in ["f","m","s"]:
            parameters["band"]["amplitude"]["tau"][i] = 1./(1+bands["amplitude"]["tau"][i])
    if("synapse" in parameters["band"]):
        for k in ["in","out"]:
            parameters["band"]["synapse"][k]["taup"] = 1./(1+bands["synapse"][k]["taup"])
            for i in ["f","m","s"]:
                parameters["band"]["synapse"][k]["tau"][i] = 1./(1+bands["synapse"][k]["tau"][i])
    parameters["freq"] = cp.deepcopy(freq) if freq is not None else {}
    for k in parameters["freq"]:
        parameters["freq"][k]["freq"] = 1./parameters["freq"][k]["period"]
        del parameters["freq"][k]["period"]
        parameters["freq"][k]["tau"] = 1./(1+parameters["freq"][k]["tau"])
    parameters["SOM"] = SOM

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

# --- top of network.py (or wherever this code lives) ---

def default_activation(u):
    """
    Default activation: sum all compartment inputs, split into E (positive)
    and I (negative) effective components.
    """
    # assume u is dict {compartment_id: tensor}
    any_comp = next(iter(u[0].values()))
    s = any_comp.clone()
    s[:] = 0
    Ieff = s.clone()
    Eff = s.clone()
    for i in u[0]:
        s += u[0][i]
        if u[1][i] > 0:
            Eff += u[0][i]
        else:
            Ieff += u[0][i]
    return s, Eff, Ieff

# for a lognormal distribution the c-th part of the mean, and coefficient of variance cv has the following probability quantile
def ln_quantile_estimator(c,cv):
    return 0.5*math.erfc(-(np.log(c)+0.5*np.log(1+cv*cv))/np.sqrt(np.log(1+cv*cv)*2))

def inverse_ln_quantile_estimator(q,cv):
    return np.exp(-erfcinv(2*q)*np.sqrt(2*np.log(1+cv*cv))/np.sqrt(1+cv*cv))