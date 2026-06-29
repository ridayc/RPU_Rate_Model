import numpy as np
import h5py


class NetworkConnectivity:
    """
    Static in-memory representation of network connectivity.

    Responsibilities:
    - load population metadata
    - load compartment metadata
    - load synapse index mappings
    - provide neuron indexing utilities
    """

    # ============================================================
    # Construction
    # ============================================================

    def __init__(self, path):
        self._path = path

        self._pop_meta = {}    # pop_id -> {size, nneu}
        self._comp_meta = {}   # pop_id -> comp_id -> metadata
        self._inds = {}        # pop_id -> comp_id -> source indices
        self._indt = {}        # pop_id -> comp_id -> target indices
        self._cached_sort = {} # pop_id -> comp_id -> cached source indices etc for sort

        self._load()

    def _load(self):
        with h5py.File(self._path, "r") as f:

            # ----------------------------------------------------
            # Populations
            # ----------------------------------------------------
            for pid in f["populations"]:
                grp = f[f"populations/{pid}"]

                self._pop_meta[pid] = {
                    "size": tuple(grp.attrs["size"]),
                    "nneu": int(grp.attrs["nneu"]),
                }

            # ----------------------------------------------------
            # Compartments
            # ----------------------------------------------------
            for pid in f["compartments"]:
                self._comp_meta[pid] = {}
                self._inds[pid] = {}
                self._indt[pid] = {}
                self._cached_sort[pid] = {}

                for cid in f[f"compartments/{pid}"]:
                    grp = f[f"compartments/{pid}/{cid}"]

                    self._comp_meta[pid][cid] = {
                        "source": str(grp.attrs["source"]),
                        "target": str(grp.attrs["target"]),
                        "k": int(grp.attrs["k"]),
                        "nsyn": int(grp.attrs["nsyn"]),
                        "type": int(grp.attrs["type"]),
                    }

                    self._inds[pid][cid] = grp["inds"][:]
                    self._indt[pid][cid] = grp["indt"][:]
                    self._cached_sort[pid][cid] = {}

    # ============================================================
    # Population API
    # ============================================================

    def population_ids(self):
        return list(self._pop_meta.keys())

    def population_size(self, population_id):
        return self._pop_meta[population_id]["size"]

    def population_nneu(self, population_id):
        return self._pop_meta[population_id]["nneu"]

    # ============================================================
    # Compartment API
    # ============================================================

    def compartment_ids(self, population_id):
        return list(self._comp_meta.get(population_id, {}).keys())

    def compartment_meta(self, population_id, comp_id):
        return self._comp_meta[population_id][comp_id]

    def all_compartments(self):
        return [
            (pid, cid)
            for pid, comps in self._comp_meta.items()
            for cid in comps
        ]

    def compartments_onto(self, population_id):
        return [(population_id, cid)
                for cid in self._comp_meta.get(population_id, {})]

    def compartments_from(self, source_population_id):
        out = []
        for pid, comps in self._comp_meta.items():
            for cid, meta in comps.items():
                if meta["source"] == source_population_id:
                    out.append((pid, cid))
        return out

    def compartment_source_population(self, population_id, comp_id):
        return self._comp_meta[population_id][comp_id]["source"]

    def compartment_target_population(self, population_id, comp_id):
        return self._comp_meta[population_id][comp_id]["target"]

    # ============================================================
    # Synapse API
    # ============================================================

    # this only need to be run once for efficient reverse source neuron searches
    def precompute_source_sort(self,population_id,comp_id):
        if("starts" not in self._cached_sort[population_id][comp_id]):
            sort_ind = np.argsort(self._inds[population_id][comp_id])
            inds_sorted = self._inds[population_id][comp_id][sort_ind]
            unique_vals, start_indices = np.unique(inds_sorted, return_index=True)
            end_indices = np.append(start_indices[1:], len(inds_sorted))
            starts = np.full(self._pop_meta[population_id]["nneu"],-1)
            starts[unique_vals] = start_indices
            ends = np.full(self._pop_meta[population_id]["nneu"],-1)
            ends[unique_vals] = end_indices
            
            self._cached_sort[population_id][comp_id] = {
                'sort_ind': sort_ind,
                'inds_sorted': inds_sorted,
                'starts': starts,
                'ends': ends
                }

    def inds(self, population_id, comp_id):
        return self._inds[population_id][comp_id]

    def indt(self, population_id, comp_id):
        return self._indt[population_id][comp_id]

    # target_idx needs to be < nneu of the target pop
    def synapse_indices_for_target(self, population_id, comp_id, target_idx):
        k = self._comp_meta[population_id][comp_id]["k"]
        
        # Handle single int - returns 2D slice
        if isinstance(target_idx, (int, np.integer)):
            start = target_idx * k
            return slice(start, start + k)  # Still works with indexing
        
        # Handle list or array - return 2D array of indices
        target_idx = np.asarray(target_idx).reshape(-1, 1)  # Column vector
        starts = target_idx * k
        # Create array of shape (n_neurons, k) with consecutive indices
        indices = starts + np.arange(k)
        return indices
    
    def source_neurons_for_target(self, population_id, comp_id, target_idx):
        k = self._comp_meta[population_id][comp_id]["k"]
        inds = self._inds[population_id][comp_id]
        
        # Handle single int
        if isinstance(target_idx, (int, np.integer)):
            start = target_idx * k
            return inds[start:start+k]  # 1D array of source neurons for that target
        
        # Handle list or array - return 2D array
        target_idx = np.asarray(target_idx).reshape(-1, 1)
        starts = target_idx * k
        # This is trickier because you need to index into indt for each start
        indices = starts + np.arange(k)
        return inds[indices]  # Returns shape (n_neurons, k)

    # source_idx needs to be < nneu of the target pop
    def synapse_indices_for_source(self, population_id, comp_id, source_idx):
        # quick source lookup only if precompute_source_sort was run for this compartment 
        cache = self._cached_sort[population_id][comp_id]
        if("starts" in cache):
            start = cache["starts"][source_idx]
            if start == -1:
                return np.array([], dtype=int)
            return cache['inds_sorted'][start:cache['ends'][source_idx]]
        else:
            return np.where(self._inds[population_id][comp_id] == source_idx)[0]

    def target_neurons_for_source(self, population_id, comp_id, source_idx):
        return self._inds[population_id][comp_id][self.synapse_indices_for_source(population_id,comp_id,source_idx)]

    # ============================================================
    # Index mapping (spatial layout)
    # ============================================================

    # get the neuron index in the population for a 3D coordinate
    # takes 3 scalars or equally sized numpy arrays as input
    def neuron_index(self, population_id, x,y,z):
        _, H, Z = self._pop_meta[population_id]["size"]
        flat_x = np.asarray(x)
        flat_y = np.asarray(y)
        flat_z = np.asarray(z)
        return flat_x * (H * Z) + flat_y * Z + flat_z

    # get the 3D coordinates from a linearize neuron index
    # accepts scalars or numpy arrays
    def neuron_xyz(self, population_id, flat_index):
        W, H, Z = self._pop_meta[population_id]["size"]

        flat_index = np.asarray(flat_index)

        x = flat_index // (H * Z)
        y = (flat_index % (H * Z)) // Z
        z = flat_index % Z

        return x, y, z

    # get the flat indices of all z values of for the input x, y coordinates
    # takes 2 scalars or equally sized numpy arrays as input
    def neurons_at_xy(self, population_id, x, y):
        _, _, Z = self._pop_meta[population_id]["size"]
        
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        
        # If scalars, treat as 1-element arrays
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
            y_arr = y_arr.reshape(1)
        
        # Create index array for all z
        z_arr = np.arange(Z)
        
        # Repeat each (x,y) for all z
        x_expanded = np.repeat(x_arr[:, None], Z, axis=1)
        y_expanded = np.repeat(y_arr[:, None], Z, axis=1)
        z_expanded = np.tile(z_arr, (len(x_arr), 1))
        
        return self.neuron_index(population_id, x_expanded, y_expanded, z_expanded)

    def all_neurons_at_xy(self, population_id):
        X, Y,_ = self._pop_meta[population_id]["size"]
        
        xs = np.arange(W).repeat(H)
        ys = np.tile(np.arange(H), W)

        return self.neurons_at_xy(population_id,xs,ys)

    def flat_xy_to_coords(self, population_id, flat_xy_index):
        """Convert flat XY index to (x, y) coordinates.
        
        The flat XY index assumes row-major order: x varies slowest, y varies fastest.
        
        Args:
            population_id: Population identifier
            flat_xy_index: Integer or array of integers, where each value is 
                        in range [0, W*H)
        
        Returns:
            Tuple of (x_coords, y_coords) as arrays
        """
        W, H, _ = self._pop_meta[population_id]["size"]
        
        flat_xy = np.asarray(flat_xy_index)
        
        x = flat_xy // H
        y = flat_xy % H
        
        return x, y

    # get  all 3D coordinates for all population indices
    def all_xyz(self, population_id):
        W, H, Z = self._pop_meta[population_id]["size"]

        xs = np.arange(W).repeat(H * Z)
        ys = np.tile(np.arange(H).repeat(Z), W)
        zs = np.tile(np.arange(Z), W * H)

        return xs,ys,zs

    def empty_xyz(self,population_id):
        W, H, Z = self._pop_meta[population_id]["size"]
        return np.zeros((W,H,Z))

    def reshape_spatial_data(self, population_id, data):
        W, H, Z = self._pop_meta[population_id]["size"]

        if data.ndim == 1:
            has_time = False
            array_length = data.shape[0]
        elif data.ndim == 2:
            has_time = True
            timesteps, array_length = data.shape
        else:
            raise ValueError(
                f"Expected 1D or 2D input array, got {data.ndim}D instead."
            )

        expected_size = W * H * Z
        if array_length != expected_size:
            raise ValueError(
                f"Array length ({array_length}) does not match "
                f"W*H*Z = {expected_size}"
            )

        if has_time:
            # (T, W, H, Z) -> (T, Z, W, H)
            return data.reshape(timesteps, W, H, Z).transpose(0, 3, 1, 2)
        else:
            # (W, H, Z) -> (Z, W, H)
            return data.reshape(W, H, Z).transpose(2, 0, 1)
