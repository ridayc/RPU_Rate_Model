import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TypeVar

from file_context import DataHandle


class DataView:
    """
    Composes file contexts and connectivity for efficient data access.
    
    All contexts are optional - only needed if the corresponding view method is called.
    
    Parameters
    ----------
    rate_ctx : RateFileContext, optional
        Context for rate files (population activity over time)
    struct_ctx : StructureFileContext, optional
        Context for structure files (synaptic weights, delays, etc.)
    connectivity : NetworkConnectivity, optional
        Connectivity reader for neuron indexing and synapse mappings
    """
    
    def __init__(self, rate_ctx=None, struct_ctx=None, connectivity=None):
        self._rate_ctx = rate_ctx
        self._struct_ctx = struct_ctx
        self._conn = connectivity

    def set_rate_ctx(self,rate_ctx):
        self._rate_ctx = rate_ctx

    def set_struct_ctx(self,struct_ctx):
        self._struct_ctx = struct_ctx

    def set_connectivity(self,connectivity):
        self._conn = connectivity        
    
    # ============================================================
    # Rate-based views (no connectivity needed)
    # ============================================================
    
    def get_rates(self, population_id: str, 
                  time_slice: Optional[Union[int, slice]] = None,
                  indices: Optional[Union[int, List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Get rates for a population.
        
        Returns shape:
        - (n_neurons,) if time_slice is int and neurons is None
        - (n_time, n_neurons) if time_slice is slice/array and neurons is None
        - (n_time, len(neurons)) if neurons specified
        """
        if self._rate_ctx is None:
            raise ValueError("Rate file context not provided")
        
        # Create handle
        handle = DataHandle("rate", population_id)
        
        # Read full time slice
        if time_slice is None:
            data = self._rate_ctx.read_slices(handle)
        else:
            data = self._rate_ctx.read_slices(handle, time_slice)
        
        # Subset neurons if requested
        return self.data_slice(data,indices)

    # ============================================================
    # Structure/parameter views
    # ============================================================
    
    def get_structure(self, population_id: str, comp_id: str, field: str,
                    time_slice: Optional[Union[int, slice]] = None,
                    indices: Optional[Union[int, List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Get synaptic weights from structure file.
        """
        if self._struct_ctx is None:
            raise ValueError("Structure file context not provided")
        
        handle = DataHandle("structure", population_id, comp_id, field)
        
        if time_slice is None:
            data = self._struct_ctx.read_slices(handle)
        else:
            data = self._struct_ctx.read_slices(handle, time_slice)
        
        return self.data_slice(data,indices)

    # ============================================================
    # Slices & Indicing
    # ============================================================

    def data_slice(self,data,indices=None):
        if indices is not None:
            indices = np.asarray(indices)
            if data.ndim == 1:
                return data[indices]
            else:
                return data[:, indices]
        return data

    # functions that require connectivity context
    def get_xy_stack(self,population_id: str, x: Union[int, List[int], np.ndarray],
                     y: Union[int, List[int], np.ndarray]) -> np.ndarray:
        if self._conn is None:
            raise ValueError("Connectivity required for xy_slicing")
        return self._conn.neurons_at_xy(population_id, x, y)  # shape (len(x), Z)

    def get_all_xy_stack(self,population_id: str) -> np.ndarray:
        if self._conn is None:
            raise ValueError("Connectivity required for xy_slicing")
        return self._conn.all_neurons_at_xy(population_id)  # shape (len(x), Z)

    def reshape_spatial_data(self,population_id: str,x: Union[List[int], np.ndarray]) -> np.array:
        if self._conn is None:
            raise ValueError("Connectivity required reshaping")
        return self._conn.reshape_spatial_data(population_id,x)

    def empty_xyz(self,population_id: str) -> np.array:
        if self._conn is None:
            raise ValueError("Connectivity population sizes")
        return self._conn.empty_xyz(population_id)

    def unravel_index(self,population_id: str,x: Union[int, List[int], np.ndarray]) -> np.array:
        if self._conn is None:
            raise ValueError("Connectivity population sizes")
        return self._conn.neuron_xyz(population_id,x)
    


    # ============================================================
    # Connectivity-based views
    # ============================================================

    def empty_xyz(self,population_id: str) -> np.array:
        if self._conn is None:
            raise ValueError("Connectivity required for population sizes")
        return self._conn.empty_xyz(population_id)

    def get_pop_size(self,population_id: str) -> np.array:
        if self._conn is None:
            raise ValueError("Connectivity required for population sizes")
        return self._conn.population_size(population_id)

    def compartment_source_population(self, population_id, comp_id):
        if self._conn is None:
            raise ValueError("Connectivity required for population sizes")
        return self._conn.compartment_source_population(population_id,comp_id)

    def compartment_target_population(self, population_id, comp_id):
        if self._conn is None:
            raise ValueError("Connectivity required for population sizes")
        return self._conn.compartment_target_population(population_id,comp_id)



    # This is a powerful data chunk reading function. It require a definition of the access function, but the access function can be used to let chunked_access work as an iterator of data streams in the view. This is one of the core compositionality tools that dataview offers for hdf5 file offers for the data stored here (it is adapted to the neuronal time series data stored here)
    def chunked_access(self,
        indices: Union[int, slice, np.ndarray],
        access_func: Callable[[Union[int, slice, np.ndarray]], np.ndarray],
        chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generic chunked data access.
        
        Args:
            indices: Time indices (int, slice, or array)
            access_func: Function that retrieves data for given indices
            chunk_size: If None, no chunking; if int, chunk into this size
        
        Returns:
            Combined data from all chunks
        """
        # Handle single index
        if isinstance(indices, (int, np.integer)):
            return access_func(indices)
        
        # Convert slice to array if chunking
        if isinstance(indices, slice):
            if chunk_size is None:
                return access_func(indices)
            # Convert to indices for chunking
            start, stop, step = indices.start or 0, indices.stop, indices.step or 1
            indices = np.arange(start, stop, step)
        
        # No chunking or small enough
        if chunk_size is None or len(indices) <= chunk_size:
            return access_func(indices)
        
        # Chunked access
        n_samples = len(indices)
        
        # Determine output shape by getting first chunk
        first_chunk = access_func(indices[:chunk_size])
        output_shape = (n_samples,) + first_chunk.shape[1:]
        result = np.empty(output_shape, dtype=first_chunk.dtype)
        result[:chunk_size] = first_chunk
        
        # Process remaining chunks
        for chunk_start in range(chunk_size, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_indices = indices[chunk_start:chunk_end]
            result[chunk_start:chunk_end] = access_func(chunk_indices)
        
        return result

    
    # ============================================================
    # Structure/parameter views
    # ============================================================

    def get_input_weights(self, target_pop: str, comp_id: str,
                       target_neuron_idx: Union[int, slice],
                       time_slice: Optional[Union[int, slice]] = None, 
                       chunks: Optional[int] = None) -> np.ndarray:
    
        if self._conn is None:
            raise ValueError("Connectivity required for source queries")
        
        syn_indices = self._conn.synapse_indices_for_target(target_pop, comp_id, target_neuron_idx)
        
        def access_weights(time_idx):
            data = self.get_structure(target_pop, comp_id, "w", time_idx)
            if isinstance(time_idx, (int, np.integer)):
                return data[syn_indices]
            return data[:, syn_indices]  # transform lives here
        
        time_slice = time_slice if time_slice is not None else slice(0,self._struct_ctx.n_written)
        return self.chunked_access(time_slice, access_weights, chunk_size=chunks)

    

    def get_source_indices(self, target_pop: str, comp_id: str,
                           target_neuron_idx: Union[int, slice],) -> np.ndarray:
        if self._conn is None:
            raise ValueError("Connectivity required for source queries")
        return self._conn.source_neurons_for_target(target_pop, comp_id, target_neuron_idx)

    
    
    # ============================================================
    # Time utilities
    # ============================================================
    
    def time_range_to_slice(self, t_start: float, t_end: float) -> slice:
        """Convert time values to slice indices using first available context."""
        ctx = self._rate_ctx or self._struct_ctx
        if ctx is None:
            raise ValueError("No file context available for time conversion")
        
        start_idx = ctx.nearest_time_index(t_start, round="down")
        end_idx = ctx.nearest_time_index(t_end, round="up") + 1
        return slice(start_idx, end_idx)
    
    def iter_time_blocks(self, population_id: str, 
                         block_size: int = 100,
                         neurons: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generator yielding time blocks for streaming processing.
        """
        if self._rate_ctx is None:
            raise ValueError("Rate file context required for iteration")
        
        n_written = self._rate_ctx.n_written
        
        for t_start in range(0, n_written, block_size):
            t_end = min(t_start + block_size, n_written)
            yield self.get_rates(population_id, slice(t_start, t_end), neurons)
    
    # ============================================================
    # Cache management
    # ============================================================
    
    def clear_cache(self):
        """Clear internal caches."""
        self._cache.clear()
    
    def cache_neuron_traces(self, population_id: str, 
                            neuron_indices: np.ndarray) -> None:
        """Pre-load and cache traces for specific neurons."""
        if self._rate_ctx is None:
            raise ValueError("Rate file context required for caching")
        
        handle = DataHandle("rate", population_id)
        full_data = self._rate_ctx.read(handle)  # (time, all_neurons)
        cache_key = (population_id, tuple(neuron_indices))
        self._cache[cache_key] = full_data[:, neuron_indices]
    
    def get_cached_traces(self, population_id: str,
                          neuron_indices: np.ndarray,
                          time_slice: Optional[slice] = None) -> Optional[np.ndarray]:
        """Retrieve cached traces if available."""
        cache_key = (population_id, tuple(neuron_indices))
        if cache_key in self._cache:
            data = self._cache[cache_key]
            if time_slice is not None:
                return data[time_slice, :]
            return data
        return None

'''
class CachedView():
    def __init__(self, datahandle, context, data):
        self._rate_ctx = rate_ctx
        self._struct_ctx = struct_ctx
        self._conn = connectivity

class ViewIterator():
    def __init__(self, datahandle, times):
        self._rate_ctx = rate_ctx
        self._struct_ctx = struct_ctx
        self._conn = connectivity
'''