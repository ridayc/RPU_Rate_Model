import argparse
import torch

from network import Network
from network_config import build_net
from data_storage import HDF5Writer, RateLogger,save_connectivity


# ============================================================
# NetworkSession
# ============================================================

class NetworkSession:
    """
    Owns a Network instance together with its optional structural
    writer (HDF5Writer) and rate logger (RateLogger), and provides
    a single step() call that handles storage timing internally.

    Designed to be used as a context manager so writers are always
    cleanly shut down:

        with build_session(args) as session:
            train_mnist(session, ...)

    Can also be constructed directly for programmatic use:

        net = build_net(device="cuda")
        session = NetworkSession(net, hdf5_every=10000, rate_every=10)
    """

    def __init__(
        self,
        net,
        writer=None,
        rate_logger=None,
        hdf5_every=10000,
        rate_every=1,
    ):
        """
        Args:
            net:          Network instance.
            writer:       HDF5Writer for structural snapshots, or None.
            rate_logger:  RateLogger for firing rate snapshots, or None.
            hdf5_every:   Store structural snapshot every N simulation timesteps.
            rate_every:   Record firing rates every N simulation timesteps.
        """
        self.net         = net
        self.writer      = writer
        self.rate_logger = rate_logger
        self.hdf5_every  = hdf5_every
        self.rate_every  = rate_every
        self._closed     = False

    def step(self):
        """
        Advance the network by one timestep, triggering structural and
        rate storage if their respective intervals are due.
        Storage checks use net.time before the iterate call, so that
        timestep 0 can trigger a snapshot of the initial state if desired.
        """
        do_hdf5  = (self.writer      is not None
                    and self.net.time % self.hdf5_every == 0)
        do_rates = (self.rate_logger is not None
                    and self.net.time % self.rate_every  == 0)
        self.net.iterate(
            writer=self.writer           if do_hdf5  else None,
            rate_logger=self.rate_logger if do_rates else None,
        )

    def set_input(self, population_id, rates):
        """
        Set firing rates for a named input population.

        Args:
            population_id:  String key into net.populations.
            rates:          1D tensor of length population.nneu.
                            Will be moved to net.device automatically.
        """
        self.net.populations[population_id].rates[:] = rates.to(self.net.device)

    def close(self):
        """Flush and shut down all storage writers. Safe to call multiple times."""
        if not self._closed:
            if self.writer:
                self.writer.close()
            if self.rate_logger:
                self.rate_logger.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# Session factory
# ============================================================

def build_session(args):
    """
    Construct a NetworkSession from a parsed argument namespace.
    Expects the namespace to contain the attributes defined by
    session_arg_parser(), plus hdf5_n_snapshots and rate_n_snapshots
    which are calculated by the caller based on total expected steps.

    Args:
        args:  argparse.Namespace from a parser that includes
               session_arg_parser() as a parent.

    Returns:
        NetworkSession instance (not yet entered as context manager).
    """
    # Load or build network
    if getattr(args, "load_snapshot", None) is not None:
        print(f"[session] Loading network from {args.load_snapshot}")
        net = Network.load(args.load_snapshot, device=args.device)
    else:
        net = build_net(
            device=args.device,
            Z_E=getattr(args, "Z_E", 16),
            frac_i=getattr(args, "frac_i", 0.28),
            scale=getattr(args, "scale", 0.1),
            active_compartments=getattr(args, "active_compartments", None),
        )

    # always write connectivity if a path is provided
    if getattr(args, "connectivity_path", None) is not None:
        save_connectivity(args.connectivity_path, net)
        print(f"[session] Connectivity saved to {args.connectivity_path}")

    # Structural HDF5 writer
    writer = None
    if getattr(args, "hdf5_path", None) is not None:
        n = getattr(args, "hdf5_n_snapshots", 1000)
        writer = HDF5Writer(args.hdf5_path, net, n)

    # Rate logger
    rate_logger = None
    if getattr(args, "rate_log_path", None) is not None:
        n      = getattr(args, "rate_n_snapshots", 1000)
        buf    = getattr(args, "rate_buffer_steps", 1000)
        rate_logger = RateLogger(args.rate_log_path, net,
                                 buffer_steps=buf, total_steps=n)

    # freeze state
    net.freeze = args.freeze

    return NetworkSession(
        net,
        writer=writer,
        rate_logger=rate_logger,
        hdf5_every=getattr(args, "hdf5_every", 10000),
        rate_every=getattr(args, "rate_every",  1),
    )


# ============================================================
# Base argument parser
# ============================================================

def session_arg_parser(
    hdf5_every=10000,
    rate_every=1,
    rate_buffer_steps=1000,
    snapshot_prefix="net",
):
    """
    Base argparse parser for NetworkSession construction.
    Use as a parent in caller scripts:

        parser = argparse.ArgumentParser(parents=[session_arg_parser()])

    Default values for storage intervals can be overridden per-caller:

        session_arg_parser(hdf5_every=100, rate_every=1)

    add_help=False is required for parent parsers.
    """
    parser = argparse.ArgumentParser(add_help=False)

    # --- Network construction ---
    parser.add_argument(
        "--load-snapshot", type=str, default=None,
        help="Path to .pt snapshot to resume from. "
             "If not set a new network is built."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string (cuda / cpu)."
    )
    parser.add_argument(
        "--Z-E", type=int, default=16,
        help="Number of excitatory depth layers."
    )
    parser.add_argument(
        "--frac-i", type=float, default=0.28,
        help="Inhibitory depth as fraction of excitatory depth."
    )
    parser.add_argument(
        "--scale", type=float, default=0.1,
        help="Global learning rate scale factor."
    )

    # --- Structural HDF5 storage ---
    parser.add_argument(
        "--hdf5-path", type=str, default=None,
        help="Path for structural HDF5 snapshots "
             "(weights, amplitudes, dN, dM). "
             "If not set, no structural snapshots are written."
    )
    parser.add_argument(
        "--hdf5-every", type=int, default=hdf5_every,
        help="Store structural snapshot every N simulation timesteps."
    )

    # --- Rate logging ---
    parser.add_argument(
        "--rate-log-path", type=str, default=None,
        help="Path for firing rate HDF5 log. "
             "If not set, no rate snapshots are written."
    )
    parser.add_argument(
        "--rate-every", type=int, default=rate_every,
        help="Record firing rates every N simulation timesteps."
    )
    parser.add_argument(
        "--rate-buffer-steps", type=int, default=rate_buffer_steps,
        help="GPU-side buffer depth in steps before flushing rates to disk."
    )

    parser.add_argument(
        "--connectivity-path", type=str, default=None,
        help="Path to save static network connectivity HDF5 file. "
         "Generated once at session start. Required for spatial plots."
    )

    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Disable learning and keep network parameters fixed."
    )

    return parser
