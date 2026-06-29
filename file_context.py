import h5py
import numpy as np


# ============================================================
# DataHandle
# ============================================================

class DataHandle:
    """
    Just describes what data we want.
    """

    def __init__(self, kind, population_id, comp_id=None, field=None):
        self.kind = kind
        self.population_id = population_id
        self.comp_id = comp_id
        self.field = field


# ============================================================
# Base File Context
# ============================================================

class BaseFileContext:
    """
    Handles:
    - opening HDF5 file
    - SWMR mode
    - time tracking (n_written, timesteps, dt)
    - reading datasets via DataHandle
    """

    def __init__(self, path, swmr=None):
        self._path = path
        self._file = None

        self._n_written = None
        self._timesteps = None
        self._dt = -1
        self._t_start = -1
        self._len = None

        if swmr is None:
            self._swmr = self._detect_swmr(path)
        else:
            self._swmr = swmr

    # ------------------------------------------------------------
    # file handling
    # ------------------------------------------------------------

    def open(self):
        if self._file is None:
            self._file = h5py.File(self._path, "r", swmr=self._swmr)
        return self._file

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def _detect_swmr(self, path):
        try:
            with h5py.File(path, "r", swmr=False):
                pass
            return False
        except OSError:
            return True

    # ------------------------------------------------------------
    # time handling
    # ------------------------------------------------------------

    def refresh_time(self):
        f = self.open()

        if(self._len==None):
            self._len = len(f["timesteps"])
        self._n_written = min(int(f["n_written"][()]),self._len)

        if self._dt<0 and self._n_written > 1:
            ts = f["timesteps"][0:2]
            self._dt = float(ts[1] - ts[0])
        
        if (self._t_start<0 and self._n_written>0):
            ts = f["timesteps"][0]
            if ts>=0:
                self._t_start = ts

    @property
    def n_written(self):
        if self._n_written is None:
            self.refresh_time()
        return self._n_written

    @property
    def timesteps(self):
        if self._timesteps is None:
            f = self.open()
            self._timesteps = f["timesteps"][:self.n_written]
        return self._timesteps

    @property
    def dt(self):
        if self._dt<0:
            self.refresh_time()
        return self._dt

    def time(self, i):
        if self._timesteps is None:
            self.timesteps  # Force load
        return self._timesteps[i]


    # ------------------------------------------------------------
    # read data
    # ------------------------------------------------------------

    # get entries from data handle object (with field for Structure Files)
    # With t as scalar: returns (n_neurons,)
    # With t as slice/array: returns (len(t), n_neurons)
    def read_slices(self, dh, t=None):
        f = self.open()
        if t is not None:
            return f[self.resolve(dh)][t,:]
        return f[self.resolve(dh)][:self.n_written,:]

    def get_len(self,dh):
        f = self.open()
        return f[self.resolve(dh)].shape[1]

    def refresh(self):
        f = self.open()

        # refresh metadata state of datasets you care about
        f["n_written"].id.refresh()

        if self._len is None:
            self._len = len(f["timesteps"])
        self._n_written = min(int(f["n_written"][()]),self._len)

    # ------------------------------------------------------------
    # time and slice calculations
    # ------------------------------------------------------------

    def nearest_time_index(self, time, round="closest"):
        timesteps = self.timesteps
        
        # Edge Case 1: Empty array protection
        if timesteps.size == 0:
            raise ValueError("Timesteps array is empty.")
            
        # Edge Case 2: Target is completely before the start
        if time <= timesteps[0]:
            return 0
            
        # Edge Case 3: Target is completely after the end
        if time >= timesteps[-1]:
            return len(timesteps) - 1

        # Base search: Find the first index where timesteps[idx] >= time
        idx = np.searchsorted(timesteps, time)
        
        # Exact match check: If it's a perfect hit, rounding direction doesn't matter
        if timesteps[idx] == time:
            return idx

        if round == "up":
            # np.searchsorted naturally points to the first element >= time.
            # Since we already ruled out exact match, timesteps[idx] is strictly > time.
            return idx
        elif round == "down":
            # The element before idx is the closest one below our target.
            return idx - 1
        elif round == "closest":
            # Bonus: True nearest time by absolute difference
            if abs(timesteps[idx] - time) < abs(timesteps[idx - 1] - time):
                return idx
            return idx - 1
        else:
            raise ValueError("Round argument must be 'up', 'down', or 'closest'")

    def bounded_time_indices(self,start,end):
        if(start>self.n_written):
            return None
        elif(end>self.n_written):
            end = self.n_written
        return [start,end]


# ============================================================
# Rate file
# ============================================================

class RateFileContext(BaseFileContext):

    def resolve(self, handle):
        return handle.population_id

    def populations(self):
        f = self.open()
        return [
            k for k in f.keys()
            if k not in ("timesteps", "n_written")
        ]

    def population_size(self, population_id):
        f = self.open()
        return f[population_id].shape[1]



# ============================================================
# Structure file
# ============================================================

class StructureFileContext(BaseFileContext):

    def resolve(self, handle):
        if handle.comp_id is None or handle.field is None:
            raise ValueError("structure handle needs comp_id and field")

        return (
                handle.population_id
                + "/"
                + handle.comp_id
                + "/"
                + handle.field
        )

    def populations(self):
        f = self.open()
        return list(f["compartments"].keys())

    def compartments(self, population_id):
        f = self.open()
        return list(f["compartments"][population_id].keys())

    def fields(self, population_id, comp_id):
        f = self.open()
        return list(f["compartments"][population_id][comp_id].keys())

    