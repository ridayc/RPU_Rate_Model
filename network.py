import torch
import math
import numpy as np
from scipy.special import erfcinv
import copy as cp

# Main class of the network
# All network populations as well as compartments (population connectivity modules) and all system wide parameters
class Network:
    def __init__(self,device,populations_param,compartments_param):
        # device for pytorch reference
        self.device = device
        # time step tracker
        # Is assumed to be an integer for storage and never below 0
        self.time = 0
        self.freeze = False
        self.barrier_event = torch.cuda.Event()

        # list of neuron populations. Python dictionaries mean that the populations must have unique identifiers (Since there is no try catch here, old populations with same name will be silently dropped (considering adding a message for this, as this is low cost here))
        # please don't call any of the populations timesteps... otherwise there will be confusion when saving the rate and structure data
        self.populations = {}
        for v in populations_param.values():
            self.populations[v["id"]] = Population(v,self)
        # compartments are created based on the user set compartment parameters and always stored in a dictionary in the target neuron population. Compartments are unique, but there can (hypothetically) be multiple different compartments between the same two populations
        for v in compartments_param.values():
            c = Compartment(v,self)
            c.target.add_compartment(c)
        # SST neurons potentially link signals from different compartments on the same target neurons. For this reason SST neurons need to set target compartment links after all compartments have been instanciated.
        for pop in self.populations.values():
            for comp in pop.compartments.values():
                if comp.SST is not None:
                    comp.SST.setup(comp)
        
    def __getstate__(self):
        """Tell pickle what to save."""
        state = self.__dict__.copy()
        if 'barrier_event' in state:
            state['barrier_event'] = None
        return state

    def __setstate__(self, state):
        """Tell pickle how to restore."""
        self.__dict__.update(state)
        # Re-initialize a fresh event upon loading
        import torch
        self.barrier_event = torch.cuda.Event()

    # iterate contains the network update for every timestep
    def iterate(self,writer=None,rate_logger=None):
        # 0. weight, amplitude, dM, dN (writer) and rate (rate_logger) storage in h5 files
        # 1. Calculate local crosscorrelation and temporal and spatial coefficients for different types of commonly inhibitory amplitude learning machinery
        # 2. Update all compartment weights based on current pre-/post weights and firing rates
        # 3. Update all compartment activations. The the activations are up to date for the rate updates
        # 4. Update all population firing rates (using the updated weights)
        # 4. Increment timestep
        # 1. and 2. are compartment wide synapse, neuron and other state updates. Assumes rates are fixed Hence compartment streams.
        # 3. Is population wide and requires all synapse weights to be stable (compartment streams ended). Population streams should end before the next iteration (hence the final synchronize)
        # All functions that are at this level need to fit into this schema, otherwise streams can break state consistency.
           
        for pop in self.populations.values():
            for comp in pop.compartments.values():
                # to use streams here we assume that each compartment updates all values using only information present in the compartment or cell that will not be updated by another compartment (that's what independent means, but any changes to code here need to be checked for this)
                with torch.cuda.stream(comp.stream):
                    if (comp.stat):
                        # different correlation statistic measurement. Currenty only used for observation purposes
                        comp.mixing_update()
                    if(not self.freeze): 
                        # selective power spectrum and correlation based measurement used for learning rules (synapses or amplitudes) which rely on different power bands 
                        comp.band_update()
                        # synaptic and amplitude weight updates per compartment
                        comp.update_weights()
                    comp.local_rate()
        barrier_stream = next(iter(self.populations.values())).stream
        for comp in self.iter_compartments():
            barrier_stream.wait_stream(comp.stream)
        
        self.barrier_event.record(barrier_stream)

        for pop in self.populations.values():
            pop.stream.wait_event(self.barrier_event)
            for comp in pop.compartments.values():
                with torch.cuda.stream(comp.stream):
                    # activations are per compartment and depend on the old rates, so they should be processed in the compartment before and parallel rate updates could happen
                    comp.update_lrates()
                    if(not self.freeze):
                        comp.update_averages()
        
        for pop in self.populations.values():
            # before updating activations and rates we need to make sure that all compartments into the population have completed their weights updates
            for comp in pop.compartments.values():
                pop.stream.wait_stream(comp.stream)
            if writer is not None:
                pop.store_snapshot_async(writer.cpu_buffers)
            # we assume that activations depend on input compartments, and that the rate update purely depends on the resulting cell wide activation. Changing this might require a review of the stream procedure here
            with torch.cuda.stream(pop.stream):
                # activations essentially assembles the local input rate accumulators
                pop.activations()
                pop.update_rates()

            # potential memory dumps for rates or rates should come here 
            # ...
            # afterwards we want to guarantee for the next iteration that all streams are synchronized again. We consider letting population streams weave into the next iteration and weight updates, but there are a lot of potentially messy scenarios that might create debugging headaches.
        torch.cuda.synchronize()
        if writer is not None:
            writer.write(self.time)
        if rate_logger is not None:
            rate_logger.record(self)
        # increase the iteration counter
        self.time+=1

    # quick function to generate an iterator over all compartments in all populations
    def iter_compartments(self):
        for p in self.populations.values():
            for c in p.compartments.values():
                yield c

    # Create a full snapshot of the whole network state on whatever device it’s currently on. This can generate a large file.
    # using pickle means that network might not be reconstructable for large scale updates object storage in the network code.
    def save(self, path: str):
        torch.save(self, path, pickle_protocol=4)

    # load that works on networks stored with save
    @staticmethod
    def load(path: str, device=None):
        # Remap *all* tensors to CPU/GPU on load (works for nested dicts too)
        if device is None:
            net = torch.load(path)  # original devices
        else:
            net = torch.load(path,map_location=device)
            net.device = device
        return net


# the neuron population class
# The variables stored here are neuron wide in the respective populations and not restricted compartments
class Population:
    def __init__(self,population_param,network):
        # back reference to the network itself. This is to keep a reference to network timestep and device without keeping a population copy
        self.net = network
        # compartments (population level connectivity) that are formed onto this population
        self.compartments = {}
        # unique population identifier. Relevant for target, source relationships of compartments
        self.id = population_param["id"]
        # commonly a 3D grid structure the neurons are assumed to live on, but this can be generically altered to N-D if desired
        # N-D however would create a less biological connectivity mapping, and the current connectivity creation approaches assume a 3D structure will be created
        # But N-D could be an interesting future direction to expand in with low cost (but this comes at the cost of untethering from a 3-D space universe...)
        self.size = cp.deepcopy(population_param["size"])
        # total number of neurons in the population
        self.nneu = math.prod(self.size)
        # stiffness of the leaky RePU (rectified polynomial unit) neuron
        self.tau = population_param["tau"]
        # polynomial exponent of the neurons in (0,inf) 
        self.p = population_param["p"]
        # characteristic response scale firing rate (at which the RePU intersects f(x)=x)
        self.r0 = population_param["r0"]
        # the minimal firing rate of population neurons (will be adjusted to be >=0 to avoid sign errors for firing rates)
        self.bias = torch.full((self.nneu,),population_param["bias"],device=self.net.device)
        # hard cap for the maximum firing rate that at which population neurons will fire
        self.cap = population_param["cap"]
        # activation function. The default assumes all compartment activations are simply added together. It is assumed that the activation will return the activation value over all 
        self.activation = population_param["activation"]
        # current firing rates of the cells
        self.rates = torch.zeros(self.nneu).to(self.net.device)
        # immediate activation function result of the current input. This is essentially smoothed into the rates (hence a leaky model)
        self.uact = torch.zeros(self.nneu).to(self.net.device)
        # combination of the compartment contributions before applying the polynomial exponent and non-linearity
        self.u_eff = torch.zeros(self.nneu).to(self.net.device)
        # the part of u_eff contributed from excitatory inputs
        self.E_eff = torch.zeros(self.nneu).to(self.net.device)
        # the part of u_eff contributed from inhibitory inputs. Values are expected to be negative
        self.I_eff = torch.zeros(self.nneu).to(self.net.device)
        # tracker of the momentary average of the firing rate of the whole population. This can be used as a proxy for SST type inhibitory neurons at a later point
        #self.ravg = torch.zeros(1).to(self.net.device)
        #self.ravg[0] = self.r0
        # tracker of the momentary average of the squared firing rate of the whole population. This can be used as a proxy for SST type inhibitory neurons at a later point
        #self.rsq = torch.zeros(1).to(self.net.device)
        #self.rsq[0] = self.r0*self.r0
        # GPU stream to run independent population operations in parallel and fill up the GPU better
        self.stream = torch.cuda.Stream()

    # add a compartment to the current population (uses a unique compartment identifier per population. Currently without try catch)
    def add_compartment(self,compartment):
        self.compartments[compartment.id] = compartment

    # combine the compartment activation contributions via the compartment activation function and apply the polynomial rectification to the result.
    def activations(self):
        u = [{},{}]
        for v in self.compartments.values():
            u[0][v.id] = v.lrates
            # the neuron types here are the sign of their rate contribution. -1 for inhibitory neurons and 1 for excitatory neurons
            u[1][v.id] = v.type
        # if there is at least one input compartment (might not be the case for input neurons) then calculate the activation
        if len(u[0])>0:
            self.activation(u,self.u_eff,self.E_eff,self.I_eff)
            self.u_eff.add_(self.bias).clamp_(min=0)
            pinv = 1/self.p
            self.uact.copy_(torch.clamp((self.u_eff**self.p)*self.r0**(1-self.p)*pinv,min=0,max=self.cap))
        # for input neurons we apply dummy values. The main reason for this is so that activation function developers don't need to handle compartments for neurons with none... But this could likely be solved more elegantly.
        else:
            self.uact.zero_()
            self.u_eff.fill_(1.)
            self.I_eff.fill_(-3.)
            self.E_eff.fill_(1.)

    # leaky rate update as well as computation of the population average of rates and squared rates
    def update_rates(self):
        #self.rates[:] = (1-self.tau)*self.rates+self.tau*self.uact
        # same as above formulation
        smoothing(self.rates,self.uact,self.tau)
        #self.ravg[0] = torch.mean(self.rates)
        #self.rsq[0] = torch.mean(self.rates*self.rates)

    # weight updates per compartment
    def update_weights(self):
        for v in self.compartments.values():
            v.update_weights()

    # power band and spectrum variable updates and measurements
    def E_I_balance(self):
        for v in self.compartments.values():
            if (v.stat):
                # different correlation statistic measurement. Currenty only used for observation purposes
                v.mixing_update()
            # selective power spectrum and correlation based measurement used for learning rules (synapses or amplitudes) which rely on different power bands 
            v.band_update()

    # full object save can't handle cuda stream objects so we work around that with a save and load state that removes the stream of this class
    def __getstate__(self):
        """Tell pickle what to save."""
        # Create a shallow copy of the object's dictionary
        state = self.__dict__.copy()
        # Remove the stream object so it doesn't block the save
        if 'stream' in state:
            state['stream'] = None 
        return state

    def __setstate__(self, state):
        """Tell pickle how to restore."""
        self.__dict__.update(state)
        # Re-initialize a fresh stream upon loading
        import torch
        self.stream = torch.cuda.Stream()

    # beware -> if this is made to write in parallel while rates are being computed, lrates would have to be removed!
    def store_snapshot_async(self, cpu_buffers):
        for comp in self.compartments.values():
            buf = cpu_buffers[self.id][comp.id]
            with torch.cuda.stream(self.stream):
                buf["w"].copy_(comp.w.view(-1), non_blocking=True)
                buf["a"].copy_(comp.a, non_blocking=True)
                buf["dN"].copy_(comp.dN, non_blocking=True)
                buf["dM"].copy_(comp.dM, non_blocking=True)
                buf["E_dw"].copy_(comp.E_dw, non_blocking=True)
                buf["E2_dw"].copy_(comp.E2_dw, non_blocking=True)
                buf["numerator"].copy_(comp.numerator, non_blocking=True)
                buf["denominator"].copy_(comp.denominator, non_blocking=True)
                buf["ravg"].copy_(comp.rate_average, non_blocking=True)
                buf["r2avg"].copy_(comp.rate_square, non_blocking=True)
                buf["rhin"].copy_(comp.rate_in, non_blocking=True)
                buf["rhout"].copy_(comp.rate_out, non_blocking=True)
                buf["wql"].copy_(comp.wql, non_blocking=True)
                buf["wqu"].copy_(comp.wqu, non_blocking=True)
                buf["corr"].copy_(comp.corr, non_blocking=True)
                if "amplitude" in comp.rate_band:
                    for band in comp.rate_band["amplitude"]["p"].keys():
                        buf[f"band_p_{band}"].copy_(comp.rate_band["amplitude"]["p"][band], non_blocking=True)


# connectivity modules connecting two neuron populations. Two populations can have multiple separate compartments in the same direction (compartment identifiers need to be unique)
class Compartment:
    def __init__(self,compartment_param,network):
        # different relevant ids
        # the target population. This compartment is stored in a dictionary in that population
        self.targetid = compartment_param["target"]
        # the source population where the input to this compartment comes from
        self.sourceid = compartment_param["source"]
        # pointers to the actual populations in the network module
        self.target = network.populations[self.targetid]
        self.source = network.populations[self.sourceid]
        # a pointer to the whole network
        self.net = network
        # unique id of this compartment (the id should be unique among the compartments in this population)
        self.id = compartment_param["id"]
        # The initial neuron amplitude parameter for all neurons. This is not used once the per neuron amplitude tensor has been created
        self.A = compartment_param["A"]
        # This is a target amplitude value that all ampltitudes are drawn towards (depends on the learning rate for this parameter)
        self.A0 = compartment_param["A0"]
        # since the amplitude targeting is learned in a logarithmic space we do the conversion and store a variable
        self.lA0 = np.log(np.abs(compartment_param["A0"]))
        # the neuron type - Inhibitory vs Excitatory is determined by the intial sign of the initial amplitude value
        self.type = np.sign(self.A)
        # later on amplitudes will be forced to be strictly positive (for log calculations etc), so we start working with positive amplitudes at this point already. Meh.
        #self.A0 = np.abs(self.A0)
        self.A*=self.type
        # This is the covariance rule ("Hebbian") learning rate
        self.eta = compartment_param["eta"]
        # learning rate of the an/ap balancing. Essentially the learning rate of how quickly ltp and ltd should become balanced on a long term average
        self.etal = compartment_param["etal"]
        # the learning rate of the shape balancing function. This essentially determines how quickly the weight distribution regularizer adjusts to the current weight distribution.
        self.etar = compartment_param["etar"]
        # below this cv value the relaxation coefficient of synaptic weight distribution will be lower (A low CV value implies the synaptic weight distribution is too narrow)
        self.thetar = compartment_param["thetar"]
        # intial relaxation rate of the synpatic weights (essentially a form of linear weight decay)
        #self.beta = compartment_param["beta"]*np.abs(self.eta)
        self.beta = compartment_param["beta"]
        # the minimal value that the beta parameter should be able to be multiplied. There is a minimal value, so that the relaxation term can not indefinitely drift towards zero
        self.beta0 = np.log(compartment_param["beta0"])
        # decay and growth exponents resp. of the asymmetric ltp/ltd rule
        self.bn = compartment_param["bn"]
        self.bp = compartment_param["bp"]
        # the quantile at which the synaptic weight distribution should be compared against the same quantile of a log normal distributoin
        self.kappa = compartment_param["kappa"]
        self.kappa2 = compartment_param["kappa2"]
        # a constant for the log normal distribution quantile location estimate
        self.zql = -np.sqrt(2)*erfcinv(2*self.kappa)
        self.zqu = np.sqrt(2)*erfcinv(2*self.kappa2)
        # learning rate of the amplitude scaling (predominantly E-E and E-I amplitudes)
        self.delta = compartment_param["delta"]
        # target ampltitude for neurons. Generally not used unless drift towards zero or infinity is expected and should be prevented
        self.rho = compartment_param["rho"]
        # double sided learning rates for special amplitude learning. Commonly the I/E input ratio for I-E amplitudes, and the correlation/spectra learning of I-I amplitudes
        self.zeta = compartment_param["zeta"]
        self.zeta2 = compartment_param["zeta2"]
        # currently cosmetic tau value that is used to track average long term firing of neurons in this compartment (intended for readout)
        self.tau = compartment_param["tau"]
        # smoothing constant for the normalization projection
        self.tauw = compartment_param["tauw"]
        # adaptation of weight smoothing based on large weight changes
        self.ck = compartment_param["ck"]
        # smoothing constant for the ltp/ltd imbalance estimation
        self.taul = compartment_param["taul"]
        # smoothing constant for estimates of gain ratios or input crosscorrelation estimates
        self.taug = compartment_param["taug"]
        # for other options than covariance learning rule. This is set to one to represent the covariance rule
        self.rin = compartment_param["rin"]
        self.rout = compartment_param["rout"]
        # this stores the averaging timescale for the covariance learning rule above. A negative sign in this value indicates that covariance rule is multiplied with input/output rate resp.
        self.tauin = compartment_param["tauin"]
        self.tauout = compartment_param["tauout"]
        self.tauout2 = compartment_param["tauout2"]
        # "Dead" synapse adjustment of the weight distribution relaxation. Quantiles of synapse weights are compared against log normal quantiles of the same CV. This value makes the distribution slightly broader by adjusting the target quantile. This introduces a small pool of noisy near zero weight synapes into the pool with the purpose of increasing cascading synaptic drift.
        self.rq = compartment_param["rq"]
        # noise injection for the initial weight distribution
        self.eps = compartment_param["eps"]
        # synapse type beyond excitatory or inhibitory. Needed for SST type and other later synapses
        self.stype = compartment_param["stype"]
        # average firing rate target for this compartment. Only relevant if self.delta is >0
        self.rate_target = compartment_param["rate_target"]
        # target value for the learning described with zeta and zeta2
        self.z_value = compartment_param["z_value"]
        # threshold value under which the z_value based learning is zero to zero
        self.thetaz = compartment_param["thetaz"]
        # string indicating the type of gain ratio for amplitude learning to be based on
        self.ratio = compartment_param["ratio"]
        # list of single compartments relevant for gain ratio amplitude learning
        self.c_c = compartment_param["c_c"]
        # default structure
        # first value if the enumerator id
        # second value is used only when the ratio of two specific compartments on the target neuron are being compared
        if(self.c_c[0]==""):
            self.c_c[0] = self.id
        self.stat = compartment_param["stat"]
        # measurment variables for compartments. Local correlation and coefficient of variance markers
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

        # ltp and ltd factors
        a0 =  np.sqrt(compartment_param["an"]*compartment_param["ap"])
        # we want the geometric mean of an and ap to be 1
        self.an = torch.full((self.target.nneu,),compartment_param["an"]*a0).to(self.net.device)
        self.ap = torch.full((self.target.nneu,),compartment_param["ap"]/a0).to(self.net.device)
        self.Jp = compartment_param["Jp"]
        self.Jn = compartment_param["Jn"]
        self.E_dw = torch.zeros(self.target.nneu).to(self.net.device)
        self.E2_dw = torch.zeros(self.target.nneu).to(self.net.device)
        self.EN_dw = torch.zeros(self.target.nneu).to(self.net.device)
        self.dM = torch.full((self.target.nneu,),np.log(compartment_param["ap"]/compartment_param["an"]),dtype=torch.float64).to(self.net.device)
        self.dN = torch.zeros(self.target.nneu,dtype=torch.float64).to(self.net.device)
        
        # quantile weight estimate based on the current CV estimate
        self.wql = torch.full((self.target.nneu,),0.).to(self.net.device)
        self.wqu = torch.full((self.target.nneu,),0.).to(self.net.device)
        # current CV estimate
        self.cv = torch.full((self.target.nneu,),0.).to(self.net.device)
        # running average and square average estimates
        self.rate_average = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        self.rate_square = torch.full((self.target.nneu,),2.*self.rate_target*self.rate_target).to(self.net.device)
        # numerator and denominators for gain ratio relations and similar
        self.numerator = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        self.denominator = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        if(self.ratio=="corr"):
            # for estimates of cross correlation of the inhibitory and the excitatory input of the neuron (might be exandable to different comparment combinations at some later point)
            self.mu_E = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
            self.mu_I = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
            self.mu2_E = torch.full((self.target.nneu,),2*self.rate_target*self.rate_target).to(self.net.device)
            self.mu2_I = torch.full((self.target.nneu,),2*self.rate_target*self.rate_target).to(self.net.device)
            self.counter = torch.zeros(self.target.nneu,dtype=torch.float).to(self.net.device)
            self.state_o = torch.zeros(self.target.nneu,dtype=torch.bool).to(self.net.device)
            self.state_n = torch.zeros(self.target.nneu,dtype=torch.bool).to(self.net.device)
            self.numerator.zero_()
        self.corr = torch.full((self.target.nneu,),0.).to(self.net.device)
        # rate average estimates for covariance learning
        self.rate_in = torch.full((self.source.nneu,),self.rate_target).to(self.net.device)
        self.rate_out = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        if(self.rout<0):
            self.rate_out2 = torch.full((self.target.nneu,),2*self.rate_target**np.abs(self.rout)).to(self.net.device)
        else:
            self.rate_out2 = torch.full((self.target.nneu,),2*self.rate_target**2).to(self.net.device)
        self.rate_in2 = torch.full((self.source.nneu,),2*self.rate_target**np.abs(self.rout)).to(self.net.device)
        # components needed for the spectral learning (primarily for I-I amplitudes, but the idea might be exandable to other power band steering behavior)
        self.band = cp.deepcopy(compartment_param["band"])
        self.rate_band = {}
        # various component to track the relative ratios of power in different frequency bands. The current setup distinguishes slow, medium and fast bands and can offer target ranges for the relative power (over all bands)
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
                for j in ["u","f","m","s"]:
                    amp[i][j] = torch.full((self.target.nneu,),0.33).to(self.net.device)

        # geometry of the input and output populations. Needed to establish initial connectivity
        target_size = torch.Size(self.target.size)
        origin_size = torch.Size(self.source.size)
        # initial synapses per target population neuron (with fixed in-degree k for efficient gpu usage; k can be smaller than the user value if the sampling region has less synapses to randomly sample from)
        points,self.k = sample_synapses(origin_size,target_size,compartment_param["ellipse"][0],compartment_param["ellipse"][1],math.prod(self.target.size),compartment_param["tsyn"])
        self.eps_w = 1e-8
        self.eps_a = 1e-6
        # total number of synapse in this compartment
        self.nsyn = self.target.nneu*self.k
        # 1. Reshape source indices into a 2D grid: (num_target_neurons, k)
        #src_grid = points.view(self.target.nneu, self.k).to(self.net.device)
        # 2. Sort along dim=1 to replicate the exact sorting of sparse .coalesce().
        # This ensures your source index lookups are clean and ordered per row.
        # We discard the sorted indices output (_) and store the values.
        self.w_ind_src, _ = torch.sort(points.view(self.target.nneu, self.k).to(self.net.device), dim=1)
        # 3. Initialize the weight matrix to match the 2D index layout
        # Shape: (num_target_neurons, k)
        # Using float32 by default; change dtype if using half-precision
        self.w = torch.zeros((self.target.nneu, self.k), dtype=torch.float32, device=self.net.device)
        # Optional: If you need to initialize weights with random values instead of zeros:
        # torch.nn.init.kaiming_uniform_(self.w, a=0, mode='fan_in')

        # temporary storage for the weight updates
        self.dw = torch.zeros((self.target.nneu, self.k), dtype=torch.float32, device=self.net.device)

        # scratch pad variables to avoid memory allocation.
        self.scratch = torch.zeros(self.target.nneu).to(self.net.device)
        self.cv_low = torch.zeros(self.target.nneu).to(self.net.device)
        self.cv_high = torch.zeros(self.target.nneu).to(self.net.device)
        self.synapse_scratch = torch.zeros((self.target.nneu, self.k)).to(self.net.device)
        self.synapse_mask = torch.zeros((self.target.nneu, self.k),dtype=torch.bool).to(self.net.device)
        #self.w+= 1./self.k
        # random initialization of the synapse weights
        self.w.copy_(torch.exp(self.eps * torch.randn_like(self.w)))
        # adjusting the covariance learning rate so that the learning parameters are targeting relational changes in the synapse weights
        self.eta/=self.k
        self.etar
        self.beta/=self.k
        # currently not in use
        #self.alpha/=self.k
        #self.thetar/=self.k
        # The current amplitude value for each neuron in the compartment. The value is strictly positive
        self.a = torch.zeros(self.target.nneu).to(self.net.device)
        self.a+=self.A
        # amplitude learning generally happens in the log space
        self.loga = torch.zeros(self.target.nneu,dtype=torch.float64).to(self.net.device)
        self.loga+=np.log(self.A)
        # input rates into the target neurons from the source population
        self.lrates = torch.zeros(self.target.nneu).to(self.net.device)
        self.lrates_buffer = torch.zeros(self.target.nneu).to(self.net.device)
        # first weight normalizatoin. This weight normalization applies to the weight tensor and weight matrix as well
        self.normalize_weights()
        # for experimental inhibitory SST neuron type
        self.SST = compartment_param["SST"]
        # GPU stream to run independent population operations in parallel and fill up the GPU better
        self.stream = torch.cuda.Stream()

    # Estimate of the incoming neuronal activation. Not really the rate yet, since all compartments per neuron need to be combined, and then the rectification and polynomial exponent are applied before the leaky "integration".
    def local_rate(self):
        '''
        Compute this compartment's contribution to the target population:
            lrates = (W @ source.rates) * a * type
            where:
              - W is row-normalized
              - a is per-target positive amplitude (homeostatic gain)
              - type is +1 (E) or -1 (I)
        '''
        if(self.SST is not None):
            self.lrates_buffer.copy_(self.SST.activation())
        else:
            self.lrates_buffer.copy_((self.weight_multiply(self.source.rates)* (self.a * self.type))) 

    def update_lrates(self):
        
        self.lrates.copy_(self.lrates_buffer)
    
    def band_update(self):
        # 
        if("amplitude" in self.rate_band):
            # frequencies should be based on firing input in a particular compartment
            self.band_power()

    # band power estimates for updates to amplitudes (generally important for I-I amplitude learning)
    def update_averages(self):
        # update the general averages
        smoothing(self.rate_average,self.target.rates,self.tau)
        smoothing(self.rate_square,self.target.rates*self.target.rates,self.tau)
        smoothing(self.rate_in,self.source.rates,np.abs(self.tauin))
        smoothing(self.rate_out,self.target.rates,np.abs(self.tauout2))
        if(self.rout<0):
            smoothing(self.rate_out2,self.target.rates**np.abs(self.rout),np.abs(self.tauout))
            smoothing(self.rate_in2,self.source.rates**np.abs(self.rout),np.abs(self.tauin))
        else:
            smoothing(self.rate_out2,self.target.rates**2,np.abs(self.tauout))


    def band_power(self):
        amp =  self.rate_band["amplitude"]
        #rates = self.target.rates
        rates = self.target.compartments[amp["target"]].lrates
        for i in amp["r"].keys():
            # ema the current rate according to the band filter timescales
            smoothing(amp["r"][i],rates,amp["tau"][i])
        for i in amp["r"].keys():
            # get the specific frequency bands for fast, mid and slow
            if(i=="u"):
                self.scratch.copy_(rates-amp["r"]["u"])
            elif(i=="f"):
                self.scratch.copy_(amp["r"]["u"]-amp["r"]["f"])
            elif(i=="m"):
                self.scratch.copy_(amp["r"]["f"]-amp["r"]["m"])
            elif(i=="s"):
                self.scratch.copy_(amp["r"]["m"]-amp["r"]["s"])
            # calculate the long term band averages
            smoothing(amp["mu"][i],self.scratch,amp["taup"])
            # calculate the long term band squared averages
            smoothing(amp["s2"][i],self.scratch*self.scratch,amp["taup"])
            # power per band is the variance of the smoothed band signal
            amp["p"][i].copy_(smooth_variance(amp["s2"][i],amp["mu"][i]))




    # correlation and CV calculations
    def mixing_update(self):
        smoothing(self.rit,self.target.rates,self.tauf)
        smoothing(self.rit_slow,self.rit,self.taus)
        smoothing(self.rjt,self.source.rates,self.tauf)
        smoothing(self.ri2t,self.target.rates*self.target.rates,self.tauf)
        smoothing(self.rj2t,self.source.rates*self.source.rates,self.tauf)
        smoothing(self.rs,self.lrates,self.tauf)
        self.r2s.copy_((self.weight_multiply(self.source.rates*self.source.rates))*self.a*self.a)
        smoothing(self.H,(self.weight_multiply(self.source.rates-self.rjt))*(self.target.rates-self.rit),self.tauf)
        self.sigi.copy_(torch.sqrt(torch.clamp(self.ri2t-self.rit*self.rit,min=0)))
        self.sigj.copy_(torch.sqrt(torch.clamp(self.rj2t-self.rjt*self.rjt,min=0)))
        self.C_fast.copy_(self.H/(self.sigi*(self.weight_multiply(self.sigj))+self.eps_a))
        smoothing(self.C,self.C_fast,self.taus)
        smoothing(self.C2,torch.abs(self.C_fast),self.taus)
        self.CVt_fast.copy_(torch.sqrt(torch.clamp(self.ri2t/(self.rit*self.rit+self.eps_a)-1,min=0)))
        smoothing(self.CVt,self.CVt_fast*self.rit,self.taus)
        self.CVs_fast.copy_(torch.sqrt(torch.clamp((self.r2s+1e-9)/(self.rs*self.rs+self.eps_a)-1,min=0)))
        smoothing(self.CVs,self.CVs_fast,self.taus)

    def amplitude_power(self):
        amp = self.rate_band["amplitude"]
        self.scratch.copy_(1./(amp["p"]["f"]+amp["p"]["m"]+amp["p"]["s"]+self.eps_a))
        Ptotinv = self.scratch

        # band frequencies dead bands for fast and slow bands
        #Regf1 = amp["p"]["f"]/(Ptot)<amp["theta"]["f"][0]
        #Regf2 = amp["p"]["f"]/(Ptot)>amp["theta"]["f"][1]
        #Regs1 = amp["p"]["s"]/(Ptot)<amp["theta"]["s"][0]
        #Regs2 = amp["p"]["s"]/(Ptot)>amp["theta"]["s"][1]

        #return Regf1*amp["eta"]["f"][0] - Regf2*amp["eta"]["f"][1] - Regs1*amp["eta"]["s"][0] + Regs2*amp["eta"]["s"][1]
        # ramp to threshold approach for the band regions
        return torch.clamp(amp["theta"]["f"][0]-amp["p"]["f"]*(Ptotinv),min=0)*amp["eta"]["f"][0]-torch.clamp(amp["p"]["f"]*(Ptotinv)-amp["theta"]["f"][1],min=0)*amp["eta"]["f"][1]-torch.clamp(amp["theta"]["s"][0]-amp["p"]["s"]*(Ptotinv),min=0)*amp["eta"]["s"][0]+torch.clamp(amp["p"]["s"]*(Ptotinv)-amp["theta"]["s"][1],min=0)*amp["eta"]["s"][1]



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
        elif(self.ratio=="supp"):
            smoothing(self.numerator,torch.clamp(self.target.compartments[self.c_c[0]].lrates+self.target.compartments[self.c_c[1]].lrates,min=0),self.taug)
            smoothing(self.denominator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
        elif(self.ratio=="sparse"):
            smoothing(self.numerator,(self.target.compartments[self.c_c[0]].lrates+self.target.compartments[self.c_c[1]].lrates)>self.thetaz,self.taug)
            self.denominator.fill_(1)
        elif(self.ratio=="thresh"):
            smoothing(self.numerator,self.target.rates>self.rate_target,self.taug)
            self.denominator.fill_(1)
        else:
            #smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            #smoothing(self.denominator,torch.abs(self.target.compartments[self.c_c[0]].lrates)+torch.abs(self.target.compartments[self.c_c[1]].lrates),self.taug)
            smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
            smoothing(self.denominator,torch.abs(self.target.compartments[self.c_c[1]].lrates),self.taug)
        return self.z_value-self.numerator/(self.denominator+self.eps_a)

    def correlation_gain(self):
        #inhibit = self.target.I_eff
        inhibit = self.lrates
        smoothing(self.mu_E,self.target.E_eff,self.taug)
        smoothing(self.mu_I,inhibit,self.taug)
        smoothing(self.mu2_E,self.target.E_eff*self.target.E_eff,self.taug)
        smoothing(self.mu2_I,inhibit*inhibit,self.taug)
        smoothing(self.corr,(self.target.E_eff-self.mu_E)*(inhibit-self.mu_I)/((torch.sqrt(torch.clamp(self.mu2_E-self.mu_E*self.mu_E,min=0))+self.eps_a)*(torch.sqrt(torch.clamp(self.mu2_I-self.mu_I*self.mu_I,min=0))+self.eps_a)),self.taug)
        return torch.abs(self.corr)-self.thetaz

    def normalize_weights(self):
        """
        Completely allocation-free L1 normalization.
        Reuses self.synapse_scratch (2D) and self.scratch (1D).
        Assumes that weights come clamped at w=self.eps_w
        """

        # 2. Sum columns directly into your 1D neuron scratchpad
        # Dropping keepdim=True lets it write directly into your 1D array
        torch.sum(self.w, dim=1, out=self.scratch)

        # 3. Clamp the 1D scratchpad in-place to avoid division by zero
        #self.scratch.clamp_(min=self.eps_w)

        # 4. Divide in-place using a zero-overhead unsqueezed view for broadcasting
        self.w.div_(self.scratch.unsqueeze(1))

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
        # reset delta w to zero

        if(self.eta!=0):
            # cov, hebbian and anti-hebbian like learning (a bit different for SST)
            if(self.SST is not None and self.SST.type!="pre"):
                self.SST.synapse()
            elif(self.SST is not None and self.SST.type=="pre"):
                self.SST.synapse()
            else:
                self.cross_rule(self.target.rates,self.source.rates,self.rate_out,self.rate_in,self.rate_out2,self.rate_in2)

            # ltp and ltd adapation for large synapses
            #self.dw.copy_(((self.a/self.A0)**(self.Jn-1)).unsqueeze(1)*torch.clamp(self.dw,max=0)*(torch.pow((1+self.k*self.w)*0.5,self.bn))+((self.a/self.A0)**(self.Jp-1)).unsqueeze(1)*torch.clamp(self.dw,min=0)*(torch.pow((1+self.k*self.w)*0.5,-self.bp)))

            
            self.synapse_scratch.copy_(self.w).mul_(self.k*0.5).add_(0.5)
            self.dw.copy_(torch.clamp(self.dw,max=0)*(torch.pow(self.synapse_scratch,self.bn))+torch.clamp(self.dw,min=0)*(torch.pow(self.synapse_scratch,-self.bp)))
            self.dw.add_(self.w)
            # clamp at zero for normalization and total mass drift estimate
            self.dw.clamp_(min=self.eps_w)

            #'''
            # prepare for an/ap adaptation to cancel of L1 mass drift effects
            if(self.etal>0 or self.etar>0):
                # estimate of long term total interal weight change average
                self.synapse_scratch.copy_(self.dw-self.w)
                smoothing(self.E2_dw,torch.sum(torch.abs(self.synapse_scratch),dim=1),self.taul)

            if(self.etal>0):
                # push an/ap ratio towards balanced ltp/ltd averages over all learning
                # estimate of long term average weight drift
                smoothing(self.E_dw,torch.sum(self.synapse_scratch,dim=1),self.taul)
                self.dM.add_((self.etal*self.E_dw/(self.E2_dw+self.eps_a)).double())
                #tM = torch.abs(self.dM.float())
                #self.ci = torch.exp(self.dM.float())
                self.ap.copy_(torch.exp(-self.dM.float()*0.5))
                self.an.copy_(torch.exp(self.dM.float()*0.5))
            #'''

            # L2 regularizer + synapse turnover (because of L1 normalization of weights the form below should have a net zero weight change)
            # that means we use this as a ltp/ltd independent weight distribution regularizer
            if(self.beta>0):
                #self.dw.add_((1/self.k*torch.sum(self.dw,dim=1).unsqueeze(1)-self.dw)*((self.beta*torch.exp(self.dN)*self.E2_dw).clamp_(max=1)).unsqueeze(1))
                self.synapse_scratch.copy_((1/self.k*torch.sum(self.dw,dim=1)).unsqueeze(1))
                # synapse_scratch = mean.unsqueeze(1) - dw, built in place: no (nneu,k) alloc
                self.synapse_scratch.sub_(self.dw)
                self.scratch.copy_((self.beta * torch.exp(self.dN.float()) * self.E2_dw).clamp_(max=1))
                # scale broadcast in place, then accumulate into dw
                self.synapse_scratch.mul_(self.scratch.unsqueeze(1))
                self.dw.add_(self.synapse_scratch)

                

            # back to smoothing and renomalization
            # two very important options here. Either we normalize and treat weight updates as a capped change of the current weights
            # Or we don't normalize and let large weight changes take full effect before normalization potentially overwriting the old weights if the rates are much higher than base line.
            # Somehow the normalized case seems to yield more realistic results. Needs more checking.
            # Currently the new weights are normalized, and the concentration of the deviations of the new weights from the old ones determines if smoothed needs to be penalized to make single weight excursions weaker.
            self.normalize_by_row(self.dw)
            self.synapse_scratch.copy_(self.dw).sub_(self.w)
            self.synapse_scratch.mul_(self.synapse_scratch)
            self.scratch.copy_(self.synapse_scratch.sum(dim=1))
            self.scratch.mul_(self.k/self.ck/self.ck).add_(1).reciprocal_().mul_(self.tauw)
            self.synapse_scratch.copy_(self.w)
            self.w.lerp_(self.dw, self.scratch.unsqueeze(1))
            self.w.clamp_(min=self.eps_w)
            self.normalize_weights()
            self.synapse_scratch.sub_(self.w).abs_()
            smoothing(self.EN_dw,0.5*torch.sum(self.synapse_scratch,dim=1),self.taul)
            #self.EN_dw.copy_(0.5*torch.sum(self.synapse_scratch,dim=1))
            
            # old idea was based on using synapse sparsity as a comparison measure. We are more explicit here. We say regularization and learning should balance each other out on short timescales (even though the net effect of the regularizer on dw is zero, the important part is the effect it has on weights at the boundary to zero!)

            # different approach. Since we assume log normal, we can estimate a lognormal wauntile from CV and use that as a live threshold. 
            #loc = "pre cv"
            #self.check_block(loc)
            
            if(self.etar>0):
                # calculate cv^2+1
                torch.mul(self.w, self.w, out=self.synapse_scratch)
                self.cv.copy_(torch.clamp(torch.sum(self.synapse_scratch,dim=1)*self.k,min=1.0+self.eps_a))
                kinv = 1./self.k
                self.scratch.copy_(torch.sqrt(torch.log(self.cv)))
                # threshold weight for the kappa quantile (based on a parametrized estimation of the log normal weight given the measured cv weight)
                self.wql.copy_(kinv/torch.sqrt(self.cv)*torch.exp(self.scratch*self.zql))
                # upper quantile threshold (reuse the same z value)
                self.wqu.copy_(kinv/torch.sqrt(self.cv) * torch.exp(self.scratch*self.zqu))
                # compare expected tail fraction to measured tail fraction.
                torch.lt(self.w, self.wql.unsqueeze(1), out=self.synapse_mask)
                self.cv_low.copy_(torch.sum(self.synapse_mask,dim=1,dtype=torch.float32)*kinv - (self.kappa + self.rq))
                # expected log normal mass
                self.cv_high.copy_(0.5*torch.erfc((self.zqu - self.scratch)/np.sqrt(2.0)))
                # upper tail mask stored in synapse-sized buffer
                torch.gt(self.w, self.wqu.unsqueeze(1), out=self.synapse_mask)
                self.synapse_scratch.copy_(self.synapse_mask)
                self.synapse_scratch.mul_(self.w)
                # M_obs*kappa2 - M_LN*C_obs
                self.cv_high.copy_(torch.sum(self.synapse_scratch,dim=1)*self.kappa2- (self.rin+1)*self.cv_high*torch.sum(self.synapse_mask,dim=1,dtype=torch.float32)*kinv)
                self.dN.add_((((self.cv<(self.thetar*self.thetar+1)).float()*-2.+torch.logical_or(self.cv_low>0,self.cv_high>0).float()-torch.logical_and(self.cv_low<0,self.cv_high<0).float())*self.EN_dw*self.etar).double())
                self.dN.clamp_(min=self.beta0)

        # adjust compartment amplitudes
        # either balance inhbition and excitation gains or balance band power fractions
        if(self.zeta!=0):
            # amplitude learning based on correlation and band power
            if(self.ratio=="corr"):
                # correlation_gain return the difference of the absolute of the average cross correlation between E and I inputs and a target value
                # if the correlation between both is too high the system is over-synchronized and the compartment amplitude should be increased.
                # Otherwise the amplitude should be adjusted to have such that its bandpower is in the proper region
                # As a rule of thumb:
                # very low I-I values leads to over-synchronization of E and I neurons. This needs to be corrected first
                # If correlations are "resonable" then too much power in the high frequency band indicates too much I-I inhibition, and too little power in the low frequency band indicates to little I-I inhibition.
                #self.loga.add_((self.zeta*torch.clamp(self.scratch,min=0)+(self.scratch<0)*self.amplitude_power()).double())
                E = self.target.compartments[self.c_c[0]]
                I = self.target.compartments[self.c_c[1]]
                self.scratch.copy_(self.correlation_gain())
                self.loga.add_((self.zeta*torch.clamp(self.scratch,min=0)+(self.scratch<0)*self.amplitude_power()).double())
            if(self.ratio=="spec"):
                self.loga.add_(self.amplitude_power().double())
            # SST neurons have some experimental slow homeostasis options. Either log-normal based distribution shaping of the target populatoin, or a PAC based theta vs gamma controller. It currently seems like standard ratio targeting of balancing the PV and SST pathways is more stable, but these options are left here in cases some one comes up with an alternative or preferable stabilization approach.Using the SST options loses control of "handtuned" SST vs PV balance, but leads to more log normal like distributions of E and/or PV population firing.
            # amplitude learning based on I-E ratios or similar
            elif(self.ratio=="EPV"):
                smoothing(self.numerator,self.lrates,self.taug)
                self.denominator.copy_((self.a+self.eps_a)/self.z_value)
                #self.loga+=(self.zeta*torch.log((self.numerator)*torch.sqrt(self.rate_target/(self.rate_average+self.eps_a)))).double()
                self.loga.add_((self.zeta*torch.log((self.numerator+self.eps_a)/self.denominator*self.rate_target/(self.rate_average+self.eps_a))).double())
            elif(self.ratio=="E2"):
                E = self.target.compartments[self.c_c[0]]
                I = self.target.compartments[self.c_c[1]]
                #self.scratch.copy_(self.rate_average<self.thetaz*E.target.cap)
                '''
                c = (self.z_value*self.z_value+1.)*self.rate_target*self.thetaz
                smoothing(self.denominator,E.lrates*E.lrates,self.taug)
                smoothing(self.numerator,E.lrates*E.a*c,self.taug)
                self.loga.add_((self.zeta*torch.log((self.numerator+self.eps_a)/(self.denominator+self.eps_a))).double())
                #'''
                #'''
                #smoothing(self.numerator,E.lrates-I.lrates>0,self.taug)
                #self.denominator.fill_(1)
                smoothing(self.numerator,E.lrates,self.taug)
                smoothing(self.denominator,E.lrates-I.lrates,self.taug)
                self.loga.add_((self.zeta*(torch.log((self.rate_average+self.eps_a)/self.rate_target)+(self.numerator/(self.denominator+self.eps_a)-self.z_value)*self.thetaz)).double())
                #'''
            elif(self.ratio=="gain"):
                self.scratch.copy_((self.a+self.eps_a)/self.z_value)
                smoothing(self.numerator,torch.abs(self.target.compartments[self.c_c[0]].lrates),self.taug)
                smoothing(self.denominator,torch.abs(self.target.compartments[self.c_c[1]].lrates),self.taug)
                self.loga.add_((self.zeta*torch.log((self.denominator*self.z_value+self.eps_a)/(self.numerator+self.eps_a))).double())
            else:
                self.loga.add_((self.zeta*self.compartment_gain()).double())
            # amplitude learning based on I-E ratios or similar
        # slow drift towards a target amplitude value
        if(self.rho>0):
            self.loga.add_((self.rho*(self.lA0-self.loga)).double())
        elif(self.rho<0):
            torch.take(torch.log(self.rate_average+self.eps_a),self.w_ind_src,out=self.synapse_scratch)
            self.loga.add_((np.sign(self.A0)*self.rho*(torch.log((self.rate_average+self.eps_a)/(torch.exp(1/self.k*torch.sum(self.synapse_scratch,dim=1))+self.eps_a))-self.lA0)).double())
        # average rate target based amplitude learning. Maybe numerical offset values should be parametrized at some point
        if(self.delta!=0):
            self.loga.add_((self.delta*torch.log((self.rate_target)/(self.rate_average+self.eps_a))).double())
        self.a.copy_(torch.exp(self.loga.float()))

        

    # full object save can't handle cuda stream objects so we work around that with a save and load state that removes the stream of this class
    def __getstate__(self):
        """Tell pickle what to save."""
        # Create a shallow copy of the object's dictionary
        state = self.__dict__.copy()
        # Remove the stream object so it doesn't block the save
        if 'stream' in state:
            state['stream'] = None 
        return state

    def __setstate__(self, state):
        """Tell pickle how to restore."""
        self.__dict__.update(state)
        # Re-initialize a fresh stream upon loading
        import torch
        self.stream = torch.cuda.Stream()

    def normalize_by_row(self, W):
        """
        W: 2D tensor of weights, shape (num_target_neurons, k)
        self.scratch: Permanent 1D tensor, shape (num_target_neurons,)
        """
        # 1. Clamp weights in-place
        W.clamp_(min=self.eps_w)
        
        # 2. Sum columns directly into your 1D scratchpad.
        # No keepdim=True here, so it perfectly fills your 1D vector.
        torch.sum(W, dim=1, out=self.scratch)
        
        # 3. Clamp the scratchpad in-place
        #self.scratch.clamp_(min=1e-8)
        
        # 4. Use unsqueeze(1) as a zero-allocation VIEW to allow broadcasting
        W.div_(self.scratch.unsqueeze(1))

    # Summing over all synapses per neuron
    def row_sum(self, value):
        """
        value: 2D tensor of shape (num_target_neurons, k)
        Returns: (num_target_neurons, 1)
        """
        return torch.sum(value, dim=1)

    # 1. Forward Pass (Completely Allocation-Free at the Synapse Level)
    def weight_multiply(self, x):
        """
        x: Source activity 1D vector of shape (num_source_neurons,)
        Returns: 1D target neuron vector of shape (num_target_neurons,)
        """
        # consider changing dw to a new synapse_scratch pad if necessary. But currently dw is only used as temporary storage for weight updates
        # Pull inputs straight into our synapse scratchpad without allocating memory
        torch.take(x, self.w_ind_src, out=self.dw)
        
        # Multiply by weights in-place (Still zero allocations)
        self.dw.mul_(self.w)
        
        # Sum along rows. This returns a clean 1D vector. 
        # (Allocating a 1D neuron vector is incredibly cheap compared to a 2D synapse matrix!)
        return torch.sum(self.dw, dim=1)

    def cross_rule(self,yi,xj,yavg,xavg,yavg2=0,xavg2=0):
        # assumes dw and synapse_scratch are not in use already!
        # bcm-like learning rule
        if(self.rout>0):
            #return (self.ap*yi*yi).unsqueeze(1)*(xj**self.rout)[self.w_ind_src]-(self.an*yi*yavg2/(yavg+self.eps_a)).unsqueeze(1)*(xj**self.rout)[self.w_ind_src]
            if(self.eta>0):
                ap = self.ap
                an = self.an
            else:
                ap = self.an
                an = self.ap
            torch.take(xj**self.rout, self.w_ind_src, out=self.synapse_scratch)
            self.dw.copy_((self.eta*yi*(ap*yi-an*yavg2/(yavg+self.eps_a))).unsqueeze(1)).mul_(self.synapse_scratch)
        # Vogels like learning rule
        elif(self.rout==0):
            # return (self.ap*yi).unsqueeze(1)*xj[self.w_ind_src]-(self.an*yavg*self.k/(torch.sum(xavg[self.w_ind_src],dim=1)+self.eps_a)).unsqueeze(1)*(xj*xavg)[self.w_ind_src]
            if(self.eta>0):
                ap = self.ap
                an = self.an
            else:
                ap = self.an
                an = self.ap
            '''    
            torch.take(xj, self.w_ind_src, out=self.dw)
            self.dw.mul_((self.eta*ap*yi*yi).unsqueeze(1))
            # denominator: sum(gather(xavg), dim=1) -- reuse synapse_scratch for the gather,
            # reduce into the existing 1D self.scratch pad (already used elsewhere for this purpose)
            torch.take(xavg, self.w_ind_src, out=self.synapse_scratch)
            torch.sum(self.synapse_scratch, dim=1, out=self.scratch)
            self.scratch.copy_(self.eta*self.an*yavg2*self.k/(self.scratch+self.eps_a))   # (nneu,) — small, not synapse-sized
            torch.take(xj * xavg, self.w_ind_src, out=self.synapse_scratch)
            self.synapse_scratch.mul_(self.scratch.unsqueeze(1))
            self.dw.sub_(self.synapse_scratch)
            #'''
            torch.take(xj, self.w_ind_src, out=self.synapse_scratch)
            self.dw.copy_((self.eta*(ap*yi*yi-an*yavg2)).unsqueeze(1)).mul_(self.synapse_scratch)
        # covariance hebb like rule
        else:
            # return (self.ap*yi**np.abs(self.rout)).unsqueeze(1)*(xj**np.abs(self.rout))[self.w_ind_src]-(self.an*yavg2).unsqueeze(1)*xavg2[self.w_ind_src]
            if(self.eta>0):
                ap = self.ap
                an = self.an
            else:
                ap = self.an
                an = self.ap
            torch.take(xj**np.abs(self.rout), self.w_ind_src, out=self.dw)
            self.dw.mul_((self.eta*ap*yi**np.abs(self.rout)).unsqueeze(1))
            torch.take(xavg2, self.w_ind_src, out=self.synapse_scratch)
            self.synapse_scratch.mul_((self.eta*an*yavg2).unsqueeze(1))
            self.dw.sub_(self.synapse_scratch)

    def check(self,loc,name, x):
        bad = ~torch.isfinite(x)
        if bad.any():
            print("Iteration: "+str(self.net.time))
            print("Compartment: "+self.id)
            print("Location: "+loc)
            print(name)
            print("NaN:", torch.isnan(x).sum().item())
            print("Inf:", torch.isinf(x).sum().item())
            print("max:", x.nan_to_num().max().item())
            print("min:", x.nan_to_num().min().item())
            print("dN_min:", self.dN.nan_to_num().min().item())
            print("dN_max:", self.dN.nan_to_num().max().item())
            raise RuntimeError(f"{name} became non-finite")

    def check_block(self,loc):
        self.check(loc,"loga", self.loga)
        self.check(loc,"lrates", self.lrates)
        self.check(loc,"I_eff", self.target.I_eff)
        self.check(loc,"E_eff", self.target.E_eff)
        self.check(loc,"u_eff", self.target.u_eff)
        self.check(loc,"rates", self.target.rates)
        self.check(loc,"dN", self.dN)
        self.check(loc,"cv", self.cv)
        self.check(loc,"dw", self.dw)
        self.check(loc,"w", self.w)
        

# EMA like sliding window
def smoothing(tracker,input,tau):
    #tracker.mul_(1 - tau).add_(tau * input)
    tracker.lerp_(input,tau)

# variance like calculation with zero clamp when zero values could be achieved
def smooth_variance(x2,x):
    return torch.clamp(x2-x*x,min=0)



# simple example of a SST neuron model (can be used for target SST neurons, or source SST neurons)
# SST neurons in this model are quite different from standard neurons
# the activation function is (currently) based off Lp normalized input activity prior to thresholding and RePU. This is an approximation or non linear dendritic gating (not of the E signal directly, but indirectly via "reweighting" of the input signal)
class SST:
    def __init__(self,sst_type,target,omega=2,tau=[1,1,1,1]):
        # p of the Lp norm
        self.omega = omega
        # target[0] is the excitatory input compartment that SST source neurons should target. The hebbian rule is adapted to take this a the post side of the hebbian rule as opposed to taking the output rate directly.
        # target[1] is used for PAC calculations and gives the compartment that the SST->? pathway that the PAC should be estimated for
        self.target = target
        # indicates if the SST neuron is the source (pre) or target neuron (post)
        self.type = sst_type
        # smoothing timescales for PAC approximations
        self.tau = cp.copy(tau)
        for i in range(len(self.tau)):
            self.tau[i] = 1./(1+self.tau[i])

    def setup(self,comp):
        self.comp = comp
        # compartment of the target population the SST should be acting on
        # we replace the target string names with the actual compartment object links at a point at which the compartments have been initialized
        # link the target compartments for the learning rules based on the unique identifier strings.
        self.target = [self.comp.target.compartments[self.target[0]],self.comp.target.compartments[self.target[1]]]
        # variables for synapse learning (depends on if the SST neuron is the source or target neuron)
        if(self.type=="pre"):
            self.g = torch.zeros(self.comp.target.nneu).to(self.comp.net.device)
            self.mu = torch.full((self.comp.target.nneu,),self.target[0].rate_target).to(self.comp.net.device)
            self.mu2 = torch.full((self.comp.target.nneu,),self.target[0].rate_target**2).to(self.comp.net.device)
        else:
            self.g = torch.zeros(self.comp.source.nneu).to(self.comp.net.device)
            self.gf = torch.full((self.comp.source.nneu,),1.).to(self.comp.net.device)
            self.gs = torch.full((self.comp.source.nneu,),1.).to(self.comp.net.device)
            self.mu = torch.full((self.comp.source.nneu,),1.).to(self.comp.net.device)
            self.mu2 = torch.full((self.comp.source.nneu,),1.).to(self.comp.net.device)

    # SST neruons have a diffent type of input effect on their targets
    # their input to a target neuron is a weighted Lp normalization of the SST input firing rates
    # this reflects that SST neurons are considered to have a more non-linear effect on the dendritic targets.
    def activation(self):
        #self.zqt_gating((self.comp.W@(self.comp.source.rates**self.omega))**(1/self.omega),self.vars)
        #return self.g*(self.comp.a*self.comp.type)
        if(self.type=="pre"):
            return self.comp.type*self.comp.a*(self.comp.weight_multiply(self.comp.source.rates**self.omega))**(1/self.omega)
        elif(self.type=="post"):
            smoothing(self.gs,self.comp.source.rates,self.tau[0])
            self.g.copy_(self.gs)
            return self.comp.type*self.comp.a*(self.comp.weight_multiply(self.gs))
        # using the SST type as a model for synaptic depression
        else:
            smoothing(self.gf,self.comp.source.rates,self.tau[0])
            self.g.copy_(torch.clamp(self.comp.source.rates-self.tau[1]*self.gf,min=0))
            smoothing(self.gs,self.g,self.tau[2])
            self.g.copy_(torch.clamp(self.g-self.tau[3]*self.gs,min=0))
            return self.comp.type*self.comp.a*(self.comp.weight_multiply(self.g))

    # synapse weight calculations are covariance hebb like. But instead of taking the direct rates, the model works with the amount of input excitation instead. The learning rate is scaled by a long term input activity average to keep learning normalized.
    def synapse(self):
        # get the 
        if(self.type=="pre"):
            self.g.copy_(self.target[0].lrates/(self.comp.a+comp.eps_a))
            tau = self.comp.tauout
            tau2 = self.comp.tauout2
            smoothing(self.mu,self.g*self.g,np.abs(tau2))
            if(self.comp.rout>=0):
                smoothing(self.mu2,self.g**2,np.abs(tau))
            else:
                smoothing(self.mu2,self.g**np.abs(self.comp.rout),np.abs(tau))
            self.comp.cross_rule(self.g,self.comp.source.rates,self.mu,self.comp.rate_in,self.mu2,self.comp.rate_in2)

        else:
            tau = self.comp.tauin
            smoothing(self.mu,self.g,np.abs(tau))
            if(self.comp.rout<0):
                smoothing(self.mu2,self.g**np.abs(self.comp.rout),np.abs(tau))
            self.comp.cross_rule(self.comp.target.rates,self.g,self.comp.rate_out,self.mu,self.comp.rate_out2,self.mu2)

    

'''
population parameters:
size: width, height and thickness of this neuron population
tau: smoothing time constant of this neuron populations firing rate
activation_exponent: exponent of the joint compartment input (0.5-1 for excitatory and 1-2 for inhibitory populations)
bias: negative threshold of the current  firing rate of the neuron (before smoothing)
cap: maximal firing rate of the neuron (after smoothing)
activation: function handle. The function should combine the local rates of each compartment in the population (by compartment id). The default in the sum of all populations. Consider using shunting populations
(eg something like:
def shunting(u):
    s = (u["excitation"]+u["inhibition"])/u["shunt_inh"]
    return s
)
'''

def population_parameters(id,size=[28,28,1],tau=0,rate_inflection = 50,activation_exponent=1,bias=0,cap=300,activation=None):
    parameters = {}
    parameters["id"] = id
    parameters["size"] = cp.copy(size)
    parameters["tau"] = 1./(1+tau)
    parameters["p"] = activation_exponent
    parameters["r0"] = rate_inflection
    parameters["cap"] = cap
    parameters["bias"] = bias
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
def compartment_parameters(id,source,target,ellipse=[1,1],tsyn=1,A=2,A0=-1,eta=0,etal=0,etar=0,alpha=0,nu=0,beta=0,beta0=1e-4,kappa=0,kappa2="",an=1,ap=1,Jn=0.5,Jp=0.5,bn=0,bp=0,c_c=None,zeta=0,zeta2="",z_value=0,thetaz=0,ratio="E/I",bands=None,rho=0,tau=0,taug=0,tauw=0,ck=1,taul=0,taur=0,thetar=0,rin=1,rout=1,tauin=1,tauout=1,tauout2="",rq=0,rt=1,noise=0,cv=0,delta=0,rate_target=1,eps=1,stype="",stat=False,power=None,freq=None,SST=None):
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
    parameters["Jn"] = Jn
    parameters["Jp"] = Jp
    parameters["kappa"] = kappa
    if(kappa2==""):
        parameters["kappa2"] = kappa
    else:
        parameters["kappa2"] = kappa2
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
    parameters["taul"] = 1./(1+taul)
    parameters["rin"] = rin
    parameters["rout"] = rout
    parameters["rq"] = rq
    parameters["rt"] = rt
    parameters["noise"] = noise
    parameters["cv"] = cv
    parameters["ck"] = ck
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
    if(tauout2==""):
        parameters["tauout2"] = parameters["tauout"]
    else:
        if(tauout2>=0):
            parameters["tauout2"] = 1./(1+tauout2)
        else:
            parameters["tauout2"] = 1./(tauout2-1)
    parameters["rate_target"] = rate_target
    parameters["band"] = cp.deepcopy(bands) if bands is not None else {}
    if("amplitude" in parameters["band"]):
        parameters["band"]["amplitude"]["taup"] = 1./(1+bands["amplitude"]["taup"])
        for i in ["u","f","m","s"]:
            parameters["band"]["amplitude"]["tau"][i] = 1./(1+bands["amplitude"]["tau"][i])
    parameters["freq"] = cp.deepcopy(freq) if freq is not None else {}
    for k in parameters["freq"]:
        parameters["freq"][k]["freq"] = 1./parameters["freq"][k]["period"]
        del parameters["freq"][k]["period"]
        parameters["freq"][k]["tau"] = 1./(1+parameters["freq"][k]["tau"])
    parameters["SST"] = SST

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
	if(a==0):
		a=1
	if(b==0):
		b=1
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

def default_activation(u,s,Eff,Ieff):
    """
    Default activation: sum all compartment inputs, split into E (positive)
    and I (negative) effective components.
    """
    # assume u is dict {compartment_id: tensor}
    s.zero_()
    Eff.zero_()
    Ieff.zero_()
    for i in u[0]:
        s.add_(u[0][i])
        if u[1][i] > 0:
            Eff.add_(u[0][i])
        else:
            Ieff.add_(u[0][i])

# for a lognormal distribution the c-th part of the mean, and coefficient of variance cv has the following probability quantile
def ln_quantile_estimator(c,cv):
    return 0.5*math.erfc(-(np.log(c)+0.5*np.log(1+cv*cv))/np.sqrt(np.log(1+cv*cv)*2))

def inverse_ln_quantile_estimator(q,cv):
    return np.exp(-erfcinv(2*q)*np.sqrt(2*np.log(1+cv*cv))/np.sqrt(1+cv*cv))