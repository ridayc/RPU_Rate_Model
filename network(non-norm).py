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
        # 1. Update all compartment weights based on current pre-/post weights
        # 2. In a second run through update all population firing rates (using the updated weights)
        # 3. Increment timestep
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
            self.uact[:] = torch.zeros(self.nneu).to(self.net.device)

    def update_rates(self):
        self.rates[:] = (1-self.tau)*self.rates+self.tau*self.uact
        self.ravg = torch.mean(self.rates).item()
        self.rsq = torch.mean(self.rates*self.rates).item()

    def update_weights(self):
        for v in self.compartments.values():
            v.update_weights()


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
        self.kappat = compartment_param["kappat"]
        self.gammat = compartment_param["gammat"]
        self.kappas = compartment_param["kappas"]
        self.gammas = compartment_param["gammas"]
        self.eps = compartment_param["eps"]
        self.M = 0
        self.rate_target = compartment_param["rate_target"]
        self.rate_average = torch.zeros(self.target.nneu).to(self.net.device)
        self.rate_average+=self.rate_target
        self.rate_square = torch.full((self.target.nneu,),self.rate_target*self.rate_target).to(self.net.device)
        self.rate_in = torch.full((self.source.nneu,),self.rate_target).to(self.net.device)
        self.rate_out = torch.full((self.target.nneu,),self.rate_target).to(self.net.device)
        self.band = cp.deepcopy(compartment_param["band"])
        self.rate_band = {}
        # create pytorch tensors to store exponential sliding windows for cosine and sine frequency bands
        for k in self.band:
            self.rate_band[k] = {"in": torch.full((self.source.nneu,),0,device=self.net.device), "out": torch.full((self.target.nneu,),0,device=self.net.device),"cin": torch.full((self.source.nneu,),0,device=self.net.device), "sin": torch.full((self.source.nneu,),0,device=self.net.device),"cout": torch.full((self.target.nneu,),0,device=self.net.device), "sout": torch.full((self.target.nneu,),0,device=self.net.device)}
        target_size = torch.Size(self.target.size)
        origin_size = torch.Size(self.source.size)
        points,self.k = sample_synapses(origin_size,target_size,compartment_param["ellipse"][0],compartment_param["ellipse"][0],math.prod(self.target.size),compartment_param["tsyn"])
        self.nsyn = self.target.nneu*self.k
        self.inds = torch.arange(self.target.nneu).repeat_interleave(self.k).to(self.net.device)
        self.indt = points.view(-1).to(self.net.device)
        self.w_ind = torch.stack((self.inds,self.indt))
        self.w = torch.zeros(self.nsyn).to(self.net.device)
        #self.w+= 1./self.k
        self.a = torch.zeros(self.target.nneu).to(self.net.device)
        self.w[:] = torch.exp(self.eps*torch.randn_like(self.w))
        self.normalize_weights()
        self.w*=self.A/self.k
        self.loga = torch.zeros(self.target.nneu).to(self.net.device)
        self.lrates = torch.zeros(self.target.nneu).to(self.net.device)
        self.W = torch.sparse_coo_tensor(self.w_ind,self.w,size=(self.target.nneu, self.source.nneu)).coalesce().to(self.net.device)
        self.w_ind[0,:] = self.W.indices()[0,:]
        self.w_ind[1,:] = self.W.indices()[1,:]
        # not needed yet, but maybe later if we have non-homogenous starting weights
        self.w[:] = self.W.values()
        self.update_weights()

    def local_rate(self):
        '''
        Compute this compartment's contribution to the target population:
            lrates = (W @ source.rates) * a * type
            where:
              - W is row-normalized
              - a is per-target positive amplitude (homeostatic gain)
              - type is +1 (E) or -1 (I)
        '''
        self.W._values()[:] = self.w
        self.lrates[:] = (self.W @ self.source.rates) *self.type

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
        for k,v in self.rate_band.items():
            freq = self.band[k]["freq"]
            tau = self.band[k]["tau"]
            cs = np.cos((self.net.time*freq)%1*2*np.pi)
            sn = np.sin((self.net.time*freq)%1*2*np.pi)
            v["in"][:] = (1-tau)*v["in"]+tau*self.source.rates
            v["out"][:] = (1-tau)*v["out"]+(tau)*self.target.rates
            v["cin"][:] = (1-tau)*v["cin"]+(tau*cs)*self.source.rates
            v["sin"][:] = (1-tau)*v["sin"]+(tau*sn)*self.source.rates
            v["cout"][:] = (1-tau)*v["cout"]+(tau*cs)*self.target.rates
            v["sout"][:] = (1-tau)*v["sout"]+(tau*sn)*self.target.rates
        # update the general averages, while we're at it
        self.rate_average[:] = (1-self.tau)*self.rate_average+self.tau*self.target.rates
        self.rate_square[:] = (1-self.tau)*self.rate_square+self.tau*self.target.rates*self.target.rates
        self.rate_in[:] = (1-np.abs(self.tauin))*self.rate_in+np.abs(self.tauin)*self.source.rates
        self.rate_out[:] = (1-np.abs(self.tauout))*self.rate_out+np.abs(self.tauout)*self.target.rates


    def normalize_weights(self):
        self.weight_amplitudes()
        norm = self.a[self.w_ind[0]]
        self.w/=(norm+1e-12)

    def weight_amplitudes(self):
        self.a.zero_()
        self.a.index_add_(0,self.w_ind[0],self.w)

    def homeostasis(self):
        # population and temporal variance forcing rates
        vart = torch.clamp(self.gammat*(self.rate_average*self.rate_average/(self.rate_square+1e-8)-self.kappat),min=0)
        vars = max(0.0,self.gammas*(self.target.ravg*self.target.ravg/(self.target.rsq+1e-8)-self.kappas))
        self.loga = self.delta*torch.log((self.rate_target+1e-3)/(self.rate_average+1e-3))+vart+vars-self.rho*torch.log((self.a+1e-3)/self.A0)
        self.loga = torch.exp(self.loga)

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
        self.homeostasis()
        # weight update, loga is already the exponentiated delta log(A) from homeostasis
        self.w*=self.loga[self.w_ind[0,:]]
        self.band_update()
        temp = torch.zeros(self.nsyn).to(self.net.device)
        temp+=self.eta
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
        self.w+=(temp+self.eta*hebb+self.nu*self.M*self.source.rates[self.w_ind[1,:]]*self.target.rates[self.w_ind[0,:]])*self.a[self.w_ind[0,:]]/self.A0-self.beta*(self.w-self.a[self.w_ind[0,:]]/self.k)
        # clamp weights
        self.w.clamp_(min=0)
        self.weight_amplitudes()

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
def compartment_parameters(id,source,target,ellipse=[1,1],tsyn=1,A=2,A0=-1,eta=0,nu=0,beta=0,bands=None,rho=0,tau=0,rin=0,rout=0,tauin=-1,tauout=-1,delta=0,gammat=0,kappat=0.5,gammas=0,kappas=0.5,rate_target=0,eps=5):
    parameters = {}
    parameters["id"] = id
    parameters["source"] = source
    parameters["target"] = target
    parameters["ellipse"] = cp.copy(ellipse)
    parameters["tsyn"] = tsyn
    parameters["A"] = A
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
        parameters["tauin"] = 1./(1+tauin)
    else:
        parameters["tauin"] = 1./(tauin-1)
    if(tauout>0):
        parameters["tauout"] = 1./(1+tauout)
    else:
        parameters["tauout"] = 1./(tauout-1)
    parameters["rate_target"] = rate_target
    parameters["gammat"] = gammat
    parameters["kappat"] = kappat
    parameters["gammas"] = gammas
    parameters["kappas"] = kappas
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
