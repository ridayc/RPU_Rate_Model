import os
import torch
import torchvision as tv
import torchvision.transforms as T

from network_session import build_session, session_arg_parser
from network_logging import log_population_stats, log_compartment_stats, log_P_E_correlation


# ============================================================
# MNIST training loop
# ============================================================

def train_mnist_unsupervised(
    session,
    epochs=1,
    steps_per_img=10,
    warm_up_per_img=5,
    seed=123,
    log_every=500,
    snapshot_dir=None,
    snapshot_every=None,
    snapshot_prefix="mnist_net",
    freeze = False
):
    """
    Unsupervised MNIST training loop.

    Each image is presented for warm_up_per_img steps (network evolves
    freely from the initial condition set by the input) followed by
    steps_per_img steps where the input is refreshed each step.
    The warm-up is intentionally free-running: P rates are set once
    before the warm-up block and not refreshed during it, allowing
    the recurrent dynamics to settle before learning steps begin.

    Args:
        session:            NetworkSession instance owning the network,
                            writer, and rate logger.
        epochs:             Number of passes through the MNIST dataset.
        steps_per_img:      Learning steps per image presentation.
        warm_up_per_img:    Free-running settling steps before learning.
        seed:               Random seed for dataset shuffling.
        log_every:          Print population/compartment stats every N images.
        snapshot_dir:       Directory for full network pickle snapshots.
                            If None, no pickle snapshots are saved.
        snapshot_every:     Save a pickle snapshot every N images.
                            Defaults to log_every if not set.
        snapshot_prefix:    Filename prefix for pickle snapshot files.
    """
    torch.manual_seed(seed)
    net = session.net
    # potentially change freeze status of network
    net.freeze = freeze

    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        if snapshot_every is None:
            snapshot_every = log_every

    transform = T.Compose([T.ToTensor()])
    trainset  = tv.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, drop_last=True, num_workers=2
    )

    P = net.populations["P"]

    img_idx_global = 0
    try:
        for epoch in range(epochs):
            for imgs, labels in loader:
                img   = imgs[0, 0] * 255.
                label = int(labels[0].item())

                # Set input once before warm-up — not refreshed during warm-up
                # so recurrent dynamics settle from this initial condition.
                #P.rates[:] = img.flatten().to(net.device)

                # Warm-up: free-running, learning still active
                for _ in range(warm_up_per_img):
                    session.step()

                # Learning steps: input refreshed each step
                for _ in range(steps_per_img):
                    P.rates[:] = img.flatten().to(net.device)
                    session.step()

                # Logging
                if img_idx_global % log_every == 0:
                    print(f"\n=== Epoch {epoch}, image {img_idx_global}, label={label} ===")
                    log_population_stats(net)
                    log_compartment_stats(net)
                    log_P_E_correlation(net)
                    print("=========================================\n")

                # Full pickle snapshot
                if (snapshot_dir and snapshot_every
                        and img_idx_global % snapshot_every == 0):
                    fname = f"{snapshot_prefix}_e{epoch}_i{img_idx_global}.pt"
                    path  = os.path.join(snapshot_dir, fname)
                    net.save(path)
                    print(f"[snapshot] Saved to {path}")

                img_idx_global += 1

    finally:
        # Clean shutdown of storage writers even on keyboard interrupt.
        # Note: session.close() is also called by the context manager in
        # the CLI block below, but calling it here ensures clean shutdown
        # when train_mnist_unsupervised is called directly without a
        # context manager.
        session.close()

        # Final pickle snapshot
        if snapshot_dir:
            fname = f"{snapshot_prefix}_final.pt"
            path  = os.path.join(snapshot_dir, fname)
            net.save(path)
            print(f"[snapshot] Saved final to {path}")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unsupervised MNIST training for recurrent E/I network.",
        parents=[session_arg_parser()],   # inherits all session/storage/device args
    )

    # MNIST-specific arguments
    parser.add_argument("--epochs",
                        type=int, default=1,
                        help="Number of passes through the MNIST dataset.")
    parser.add_argument("--steps-per-img",
                        type=int, default=10,
                        help="Learning steps per image presentation.")
    parser.add_argument("--warm-up-per-img",
                        type=int, default=5,
                        help="Free-running settling steps before learning steps.")
    parser.add_argument("--seed",
                        type=int, default=123,
                        help="Random seed.")
    parser.add_argument("--log-every",
                        type=int, default=500,
                        help="Print stats every N images.")
    parser.add_argument("--snapshot-dir",
                        type=str, default=None,
                        help="Directory for full network pickle snapshots.")
    parser.add_argument("--snapshot-every",
                        type=int, default=None,
                        help="Save pickle snapshot every N images. "
                             "Defaults to --log-every if not set.")
    parser.add_argument("--snapshot-prefix",
                        type=str, default="mnist_net",
                        help="Filename prefix for pickle snapshot files.")

    args = parser.parse_args()

    # Pre-allocate HDF5 datasets based on expected total simulation steps.
    # Done here because only the runner knows the epoch/image structure.
    transform = T.Compose([T.ToTensor()])
    trainset  = tv.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    total_steps = (args.epochs * len(trainset)
                   * (args.steps_per_img + args.warm_up_per_img))
    args.hdf5_n_snapshots = max(1, total_steps // args.hdf5_every)
    args.rate_n_snapshots = max(1, total_steps // args.rate_every)

    # Print storage size estimate before committing
    if args.hdf5_path or args.rate_log_path:
        print(f"[storage] Total simulation steps: {total_steps:,}")
        if args.hdf5_path:
            print(f"[storage] Structural snapshots:   {args.hdf5_n_snapshots:,} "
                  f"(every {args.hdf5_every} steps) -> {args.hdf5_path}")
        if args.rate_log_path:
            print(f"[storage] Rate snapshots:          {args.rate_n_snapshots:,} "
                  f"(every {args.rate_every} steps) -> {args.rate_log_path}")

    with build_session(args) as session:
        train_mnist_unsupervised(
            session,
            epochs=args.epochs,
            steps_per_img=args.steps_per_img,
            warm_up_per_img=args.warm_up_per_img,
            seed=args.seed,
            log_every=args.log_every,
            snapshot_dir=args.snapshot_dir,
            snapshot_every=args.snapshot_every,
            snapshot_prefix=args.snapshot_prefix,
        )
