import torch
import h5py
import threading
import queue
import numpy as np


# ============================================================
# CPU Buffer Management
# Pinned memory buffers for async GPU->CPU transfers.
# One buffer dict per population, one tensor per stored field
# per compartment. Pin_memory() enables non-blocking GPU copies.
# ============================================================

def initialize_cpu_buffers(network):
    """
    Allocate pinned CPU tensors matching the shape and dtype of every
    stored field for every compartment in the network.

    Returns:
        Nested dict: {pop_id: {comp_id: {field_name: pinned_tensor}}}
    """
    cpu_buffers = {}
    for pop in network.populations.values():
        cpu_buffers[pop.id] = {}
        for comp in pop.compartments.values():
            cpu_buffers[pop.id][comp.id] = {
                "w":  torch.empty(comp.nsyn,        dtype=torch.float32).pin_memory(),
                "a":  torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "dN": torch.empty(comp.target.nneu, dtype=torch.float64).pin_memory(),
                "dM": torch.empty(comp.target.nneu, dtype=torch.float64).pin_memory(),
                "E_dw": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "E2_dw": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "numerator": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "denominator": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "ravg": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "r2avg": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "rhin": torch.empty(comp.source.nneu, dtype=torch.float32).pin_memory(),
                "rhout": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "wql": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "wqu": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
                "corr": torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory(),
            }
            if "amplitude" in comp.rate_band:
                for band in ["u","f", "m", "s"]:
                    cpu_buffers[pop.id][comp.id][f"band_p_{band}"] = (
                        torch.empty(comp.target.nneu, dtype=torch.float32).pin_memory()
                    )
    return cpu_buffers


def initialize_cpu_buffers_from_existing(old_buffers):
    """
    Allocate a fresh set of pinned CPU buffers with the same shapes and
    dtypes as an existing buffer dict. Used to replace buffers that have
    been handed off to the writer thread so the GPU can immediately start
    filling new buffers without waiting for the write to complete.

    Returns:
        Nested dict with same structure as old_buffers.
    """
    new_buffers = {}
    for pop_id, pop_buffers in old_buffers.items():
        new_buffers[pop_id] = {}
        for comp_id, buffers in pop_buffers.items():
            new_buffers[pop_id][comp_id] = {
                name: torch.empty_like(tensor).pin_memory()
                for name, tensor in buffers.items()
            }
    return new_buffers


# ============================================================
# Structural Snapshot Storage
# Stores per-compartment weights, amplitudes, and slow accumulators
# (dN, dM) at configurable intervals during training.
# Files are created with libver='latest' and SWMR mode to allow
# concurrent reading while the simulation is running.
# ============================================================

def initialize_storage(file_path, network, n_snapshots):
    """
    Pre-allocate HDF5 file structure for structural snapshots.
    Must be called before HDF5Writer is constructed.
    Uses libver='latest' required for SWMR concurrent read support.
    Chunk size of (1, n_features) aligns with one snapshot per write.

    Args:
        file_path:    Output HDF5 path.
        network:      Network instance (read-only, for shapes).
        n_snapshots:  Number of snapshot slots to pre-allocate.
    """
    with h5py.File(file_path, 'w', libver='latest') as f:
        # single timestep per snapshot — read as whole array to count written
        f.create_dataset("timesteps",
                         shape=(n_snapshots,),
                         dtype='int64',
                         fillvalue=-1,
                         chunks=(n_snapshots,))
        f.create_dataset("n_written",data=0)
        for pop in network.populations.values():
            pop_grp = f.create_group(pop.id)
            for comp in pop.compartments.values():
                comp_grp = pop_grp.create_group(comp.id)
                # chunk size (1, n) aligns with one-snapshot-at-a-time writes
                comp_grp.create_dataset("w",
                    shape=(n_snapshots, comp.nsyn),
                    dtype='float32',
                    chunks=(1, comp.nsyn))
                comp_grp.create_dataset("a",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                # dN and dM stored as float64 — these are log-domain
                # accumulators where float32 precision is insufficient
                # due to timescale separation (see network.py)
                comp_grp.create_dataset("dN",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float64',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("dM",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float64',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("E_dw",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("E2_dw",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                if "amplitude" in comp.rate_band:
                    for band in ["u","f", "m", "s"]:
                        comp_grp.create_dataset(f"band_p_{band}",
                            shape=(n_snapshots, comp.target.nneu),
                            dtype='float32',
                            chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("numerator",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("denominator",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("ravg",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("r2avg",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("rhin",
                    shape=(n_snapshots, comp.source.nneu),
                    dtype='float32',
                    chunks=(1, comp.source.nneu))
                comp_grp.create_dataset("rhout",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("wql",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("wqu",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))
                comp_grp.create_dataset("corr",
                    shape=(n_snapshots, comp.target.nneu),
                    dtype='float32',
                    chunks=(1, comp.target.nneu))


class HDF5Writer:
    """
    Asynchronous structural snapshot writer.

    GPU->CPU transfers are initiated in the population stream via
    store_snapshot_async() on the Population object, overlapping with
    rate updates. After torch.cuda.synchronize() the pinned buffers are
    complete and write() hands them to this writer's background thread.
    Fresh buffers are allocated immediately so the next iteration can
    begin GPU transfers without waiting for disk I/O.
    """

    def __init__(self, file_path, network, n_snapshots):
        """
        Args:
            file_path:    Path for the structural HDF5 file.
            network:      Network instance (for buffer allocation).
            n_snapshots:  Pre-allocated snapshot slots.
        """
        self.file_path = file_path
        initialize_storage(file_path, network, n_snapshots)
        self.cpu_buffers  = initialize_cpu_buffers(network)
        self.queue        = queue.Queue()
        self.thread       = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        self.snapshot_idx = 0

    def _writer_loop(self):
        # libver='latest' required to enable SWMR after open
        with h5py.File(self.file_path, 'a', libver='latest') as f:
            f.swmr_mode = True   # enables concurrent readers
            while True:
                item = self.queue.get()
                if item is None:
                    break
                snapshot_idx, timestep, cpu_buffers = item
                f["timesteps"][snapshot_idx] = timestep
                for pop_id, pop_buffers in cpu_buffers.items():
                    for comp_id, buffers in pop_buffers.items():
                        comp_grp = f[f"{pop_id}/{comp_id}"]
                        for name, tensor in buffers.items():
                            comp_grp[name][snapshot_idx] = tensor.numpy()
                # flush commits data to disk so SWMR readers can see it
                dset = f["n_written"] 
                dset[()] = snapshot_idx+1
                f.flush()
                self.queue.task_done()

    def write(self, timestep):
        """
        Hand current CPU buffers to the writer thread and immediately
        allocate fresh buffers for the next iteration's GPU transfers.
        The old buffer reference is held by the queue until written.
        """
        self.queue.put((self.snapshot_idx, timestep, self.cpu_buffers))
        self.cpu_buffers   = initialize_cpu_buffers_from_existing(self.cpu_buffers)
        self.snapshot_idx += 1

    def close(self):
        """Drain the write queue and shut down the writer thread cleanly."""
        self.queue.join()
        self.queue.put(None)
        self.thread.join()


# ============================================================
# Connectivity Storage
# Saves static network structure once at session start.
# Contains population sizes and synapse index arrays needed for
# spatial plots without loading the full .pt network snapshot.
# Compartments are stored under their target population id since
# compartment ids are only unique per target population.
# ============================================================

def save_connectivity(file_path, net):
    """
    Save static network connectivity to HDF5.

    Both inds (target neuron indices) and indt (source neuron indices)
    are stored explicitly. Storing inds is necessary because
    torch.sparse_coo_tensor.coalesce() reorders synapse entries
    lexicographically, breaking the original repeat_interleave structure
    that would otherwise allow reconstruction from nneu and k alone.

    Args:
        file_path:  Output HDF5 path.
        net:        Network instance.
    """
    with h5py.File(file_path, 'w') as f:
        # population metadata
        for pid, pop in net.populations.items():
            grp = f.create_group(f"populations/{pid}")
            grp.attrs["size"] = pop.size
            grp.attrs["nneu"] = pop.nneu

        # compartment connectivity nested under target population id
        for pid, pop in net.populations.items():
            for cid, comp in pop.compartments.items():
                grp = f.create_group(f"compartments/{pid}/{cid}")
                grp.attrs["source"] = comp.sourceid
                grp.attrs["target"] = comp.targetid
                grp.attrs["k"]      = comp.k
                grp.attrs["nsyn"]   = comp.nsyn
                grp.attrs["type"]   = comp.type
                # both arrays stored as int64 — int32 would be safe for
                # current network sizes but int64 avoids any future limit
                grp.create_dataset("indt",
                    data=torch.arange(comp.target.nneu, device=comp.net.device).repeat_interleave(comp.k).cpu().numpy(), dtype='int64')
                grp.create_dataset("inds",
                    data=comp.w_ind_src.view(-1).cpu().numpy(), dtype='int64')


# ============================================================
# Rate Logger
# Records population firing rates at configurable intervals.
# Rates accumulate in GPU-side buffers for buffer_steps steps
# before being flushed to CPU and written to HDF5 asynchronously.
# Files use libver='latest' and SWMR for concurrent read support.
# Chunk size aligns with buffer_steps for efficient sequential writes.
# ============================================================

class RateLogger:
    """
    Asynchronous firing rate logger with GPU-side buffering.

    Rates are accumulated in GPU tensors for buffer_steps steps before
    a CPU copy is initiated. The HDF5 write happens in a background
    thread so the simulation is not blocked by disk I/O.
    """

    def __init__(self, file_path, net, buffer_steps=1000, total_steps=None):
        """
        Args:
            file_path:     Output HDF5 path.
            net:           Network instance.
            buffer_steps:  GPU buffer depth before flushing to disk.
                           Chunk size is set to match this value so
                           each flush writes exactly one chunk.
            total_steps:   Total rate snapshots to pre-allocate.
        """
        self.file_path    = file_path
        self.buffer_steps = buffer_steps
        self.step         = 0   # steps accumulated in current buffer
        self.total_written = 0  # total steps flushed to HDF5

        self.pop_ids = list(net.populations.keys())
        self.nneu    = {pid: pop.nneu for pid, pop in net.populations.items()}
        device       = net.device

        # GPU-side accumulation buffers: (buffer_steps, nneu) per population
        self.buffers = {
            pid: torch.zeros(buffer_steps, self.nneu[pid], device=device)
            for pid in self.pop_ids
        }
        # assuming that simulation timesteps never start below 0
        self.timestep_buf = torch.full((buffer_steps,), -1, dtype=torch.int64,
                                        device=device)

        # pre-allocate HDF5 with libver='latest' for SWMR support
        # chunk size matches buffer_steps so each flush writes one chunk
        with h5py.File(file_path, 'w', libver='latest') as f:
            f.create_dataset("timesteps",
                             shape=(total_steps,),
                             dtype='int64',
                             fillvalue=-1,
                             chunks=(min(buffer_steps, total_steps),))
            f.create_dataset("n_written",data=0)
            for pid in self.pop_ids:
                f.create_dataset(pid,
                    shape=(total_steps, self.nneu[pid]),
                    dtype='float32',
                    chunks=(min(buffer_steps, total_steps), self.nneu[pid]))

        self.queue  = queue.Queue()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def record(self, net):
        """
        Accumulate current firing rates into the GPU buffer.
        Call once per simulation step after torch.cuda.synchronize().
        Flushes automatically when buffer is full.
        """
        for pid, pop in net.populations.items():
            self.buffers[pid][self.step] = pop.rates
        self.timestep_buf[self.step] = net.time
        self.step += 1
        if self.step == self.buffer_steps:
            self._flush()

    def _flush(self):
        """Copy current GPU buffer slice to CPU and queue for writing."""
        if self.step == 0:
            return
        n = self.step
        cpu_rates = {
            pid: self.buffers[pid][:n].cpu()
            for pid in self.pop_ids
        }
        cpu_ts = self.timestep_buf[:n].cpu()
        self.queue.put((self.total_written, n, cpu_rates, cpu_ts))
        self.total_written += n
        self.step = 0

    def _writer_loop(self):
        # libver='latest' required to enable SWMR after open
        with h5py.File(self.file_path, 'a', libver='latest') as f:
            f.swmr_mode = True   # enables concurrent readers
            while True:
                item = self.queue.get()
                if item is None:
                    break
                start, n, cpu_rates, cpu_ts = item
                f["timesteps"][start:start+n] = cpu_ts.numpy()
                for pid in self.pop_ids:
                    f[pid][start:start+n] = cpu_rates[pid].numpy()
                # flush commits data so SWMR readers can see latest chunks
                dset = f["n_written"] 
                dset[()] = start+n
                f.flush()
                self.queue.task_done()

    def close(self):
        """Flush remaining buffer, drain queue, shut down writer thread."""
        self._flush()
        self.queue.join()
        self.queue.put(None)
        self.thread.join()