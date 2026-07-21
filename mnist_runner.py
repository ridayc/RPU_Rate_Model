import os
import torch
import torchvision as tv
import torchvision.transforms as T

from network_session import build_session, session_arg_parser
from network_logging import log_population_stats, log_compartment_stats, log_P_E_correlation
from network import smoothing


# ============================================================
# Sleep & Wake Protocol Helpers
# ============================================================

def set_population_attribute(pop, attr_name, value):
    """Safely updates a population attribute (like bias) in-place if it is a Tensor."""
    if hasattr(pop, attr_name):
        attr = getattr(pop, attr_name)
        if isinstance(attr, torch.Tensor):
            attr.copy_(value)
        else:
            setattr(pop, attr_name, value)

def get_population_attribute_copy(pop, attr_name):
    """Returns a safe detached copy or scalar value of a population attribute."""
    if hasattr(pop, attr_name):
        attr = getattr(pop, attr_name)
        if isinstance(attr, torch.Tensor):
            return attr.clone()
        return attr
    return None

def run_nrem_phase(session, target_populations, daily_mean, steps, c_bias, c_burst, up_steps, down_steps,t_spindle):
    """
    Executes NREM sleep.
    Applies a deep negative bias scaled to local daytime activity.
    Alternates between Up states (zero-clamped noise bursts) and Down states (pure silence).
    """
    net = session.net
    P = net.populations["P"]

    # Phase 1: Apply NREM hyperpolarizing bias
    orig_biases = {pop_id: get_population_attribute_copy(net.populations[pop_id], "bias") for pop_id in target_populations}
    
    for pop_id in target_populations:
        pop = net.populations[pop_id]
        # Attempt to scale bias by external input gain (A_PE) if available
        if "P_E" in pop.compartments:
            nrem_bias = -pop.compartments["P_E"].weight_multiply(daily_mean) * c_bias
        else:
            nrem_bias = -daily_mean * c_bias
            
        set_population_attribute(pop, "bias", nrem_bias)

    # Phase 2: Run the Slow Wave Oscillator
    cycle_length = up_steps + down_steps
    
    for i in range(steps):
        if (i % cycle_length) < up_steps:
            # Up State: Spindle bursts (zero-clamped gaussian noise)
            if(i%t_spindle==0):
                P.rates.copy_((daily_mean * c_burst * torch.randn_like(P.rates)).clamp_(min=0))
            else:
                P.rates.zero_()
        else:
            # Down State: Absolute silence for W_EE recovery
            P.rates.zero_()
            
        session.step()

    # Phase 3: Restore wake baseline bias
    for pop_id in target_populations:
        set_population_attribute(net.populations[pop_id], "bias", orig_biases[pop_id])


def run_rem_phase(session, daily_mean, steps, c_tonic, c_burst, rem_tonic_steps, phasic_steps, micro_burst_steps, micro_silence_steps):
    """
    Executes REM sleep.
    Bias remains at Wake baseline (I_AC = 0 relative to sleep).
    Defaults to Tonic REM (low-variance noise).
    Transitions to Phasic REM based on a structural transition rate (1 / rem_tonic_steps),
    generating an organic exponential distribution of tonic intervals without hacky jitter parameters.
    """
    net = session.net
    P = net.populations["P"]
    
    phasic_timer = 0
    micro_cycle_len = micro_burst_steps + micro_silence_steps
    
    # Calculate the clean, memoryless transition probability out of the Tonic state
    p_enter_phasic = 1.0 / float(rem_tonic_steps)

    for i in range(steps):
        # 1. State machine check: If in Tonic (timer == 0), roll for Phasic intrusion
        if phasic_timer == 0:
            if torch.rand(1).item() < p_enter_phasic:
                phasic_timer = phasic_steps
                
        # 2. Compute background Tonic noise
        #tonic_noise = torch.randn_like(P.rates) * c_tonic * daily_mean
        
        # 3. Apply phase-specific dynamics
        if phasic_timer > 0:
            # Inside a Phasic REM window. Evaluate the micro-burst rhythm.
            time_spent_in_phasic = phasic_steps - phasic_timer
            micro_step = time_spent_in_phasic % micro_cycle_len
            
            if micro_step < micro_burst_steps:
                # Active PGO micro-burst: Tonic background is completely overridden
                #burst_noise = torch.randn_like(P.rates) * c_burst * daily_mean
                P.rates.copy_((torch.randn_like(P.rates) * c_burst * daily_mean).clamp_(min=0))
            else:
                # Micro-silence between bursts: Tonic background resumes
                P.rates.copy_((torch.randn_like(P.rates) * c_tonic * daily_mean).clamp_(min=0))
                
            phasic_timer -= 1
        else:
            # Pure Tonic REM: Just the low-variance baseline
            P.rates.copy_((torch.randn_like(P.rates) * c_tonic * daily_mean).clamp_(min=0))

        session.step()


# ============================================================
# MNIST training loop
# ============================================================

def train_mnist_unsupervised(
    session,
    args,
    epochs=1,
    steps_per_img=10,
    warm_up_per_img=5,
    seed=123,
    log_every=500,
    snapshot_dir=None,
    snapshot_every=None,
    snapshot_prefix="mnist_net",
    freeze=False
):
    """
    Unsupervised MNIST training loop with macro-scale sleep/wake protocol integration.
    """
    torch.manual_seed(seed)
    net = session.net
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
    daily_mean = P.rates.clone().zero_()
    
    if args.sleep_every_n_images is not None:
        tau_d = 1.0 / (args.sleep_every_n_images + 1.0)
    else:
        tau_d = 0.0

    steps_per_cycle = warm_up_per_img + steps_per_img
    img_idx_global = 0

    try:
        for epoch in range(epochs):
            for imgs, labels in loader:
                img   = imgs[0, 0] * 255.
                label = int(labels[0].item())

                # Warm-up: free-running, learning still active
                for _ in range(warm_up_per_img):
                    # Example wake noise logic
                    P.rates.copy_(torch.rand_like(P.rates) * args.noise).clamp_(min=0)
                    if(tau_d>0):
                        smoothing(daily_mean, P.rates, tau_d)
                    session.step()

                # Learning steps: input refreshed each step
                temp_img = img.flatten().to(net.device)
                for _ in range(steps_per_img):
                    P.rates.copy_(temp_img)
                    P.rates.add_(torch.rand_like(P.rates) * args.noise).clamp_(min=0)
                    if(tau_d>0):
                        smoothing(daily_mean, P.rates, tau_d)
                    session.step()

                # Logging
                if img_idx_global % log_every == 0:
                    print(f"\n=== Epoch {epoch}, image {img_idx_global}, label={label} ===")
                    log_population_stats(net)
                    log_compartment_stats(net)
                    log_P_E_correlation(net)
                    print("=========================================\n")

                # Full pickle snapshot
                if (snapshot_dir and snapshot_every and img_idx_global % snapshot_every == 0):
                    fname = f"{snapshot_prefix}_e{epoch}_i{img_idx_global}.pt"
                    path  = os.path.join(snapshot_dir, fname)
                    net.save(path)
                    print(f"[snapshot] Saved to {path}")

                img_idx_global += 1

                # Check if it is time to trigger the Macro Sleep/Wake Cycle
                if args.sleep_every_n_images is not None and img_idx_global % args.sleep_every_n_images == 0:
                    target_pops = args.sleep_populations if args.sleep_populations else ["E"]

                    print(f"\n[Macro-Protocol] Triggering offline sleep cycle at image {img_idx_global}...")
                    
                    nrem_steps = args.sleep_cycles * steps_per_cycle
                    rem_steps  = args.rest_cycles * steps_per_cycle
                    n_img = (args.sleep_cycles + args.rest_cycles) * args.rotations
                    n_steps = n_img * steps_per_cycle

                    print(f"Starting {args.rotations} Sleep Rotations ({n_steps} total steps)")
                    
                    for i in range(args.rotations):
                        print(f"  -> Rotation {i+1}/{args.rotations}: NREM phase")
                        run_nrem_phase(
                            session=session, 
                            target_populations=target_pops, 
                            daily_mean=daily_mean, 
                            steps=nrem_steps, 
                            c_bias=args.nrem_bias_scale, 
                            c_burst=args.nrem_burst_scale, 
                            up_steps=args.nrem_up_steps, 
                            down_steps=args.nrem_down_steps,
                            t_spindle=args.t_spindle
                        )
                        
                        print(f"  -> Rotation {i+1}/{args.rotations}: REM phase")
                        run_rem_phase(
                            session=session, 
                            daily_mean=daily_mean, 
                            steps=rem_steps, 
                            c_tonic=args.rem_tonic_scale, 
                            c_burst=args.rem_burst_scale, # Reuse NREM burst scale for PGO waves
                            rem_tonic_steps=args.rem_tonic_steps, 
                            phasic_steps=args.rem_phasic_steps, 
                            micro_burst_steps=args.rem_micro_burst_steps, 
                            micro_silence_steps=args.rem_micro_silence_steps
                        )

                    print("[Macro-Protocol] Cycle complete. Resuming MNIST Wake phase.\n")

    finally:
        session.close()
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
        description="Unsupervised MNIST training for recurrent E/I network with Sleep/Wake cycles.",
        parents=[session_arg_parser()],
    )

    # MNIST-specific arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-img", type=int, default=10)
    parser.add_argument("--warm-up-per-img", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--snapshot-every", type=int, default=None)
    parser.add_argument("--snapshot-prefix", type=str, default="mnist_net")
    parser.add_argument("--noise", type=float, default=5.0)

    # Macro Cycle Configuration
    parser.add_argument("--sleep-every-n-images", type=int, default=4000, help="Images per Wake cycle.")
    parser.add_argument("--sleep-populations", type=str, nargs="+", default=["E"], help="Populations receiving sleep bias.")
    parser.add_argument("--rotations", type=int, default=2, help="Number of NREM->REM blocks per night.")
    parser.add_argument("--sleep-cycles", type=int, default=200, help="Duration of NREM phase in image steps.")
    parser.add_argument("--rest-cycles", type=int, default=300, help="Duration of REM phase in image steps.")

    # NREM Parameters
    parser.add_argument("--nrem-bias-scale", type=float, default=1., help="Multiplier for the I_AC negative clamp.")
    parser.add_argument("--nrem-burst-scale", type=float, default=10., help="Multiplier for spindle burst amplitude.")
    parser.add_argument("--nrem-up-steps", type=int, default=50, help="Duration of the spindle burst window.")
    parser.add_argument("--nrem-down-steps", type=int, default=150, help="Duration of the deep silence gap.")
    parser.add_argument("--t-spindle", type=int, default=10, help="Interval of spindle bursts during NREM up states")

    # REM Parameters
    parser.add_argument("--rem-tonic-scale", type=float, default=2.5, help="Fraction of daily_mean used for tonic noise.")
    parser.add_argument("--rem-burst-scale", type=float, default=5., help="Multiplier for pgo micro-burst amplitude.")
    parser.add_argument("--rem-tonic-steps", type=int, default=150, help="Expected duration of a Tonic interval between Phasic bursts.")
    parser.add_argument("--rem-phasic-steps", type=int, default=60, help="Total length of a Phasic REM cluster.")
    parser.add_argument("--rem-micro-burst-steps", type=int, default=1, help="Duration of a single PGO micro-burst.")
    parser.add_argument("--rem-micro-silence-steps", type=int, default=11, help="Silence gap between PGO micro-bursts.")

    args = parser.parse_args()

    transform = T.Compose([T.ToTensor()])
    trainset  = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    steps_per_image = args.steps_per_img + args.warm_up_per_img
    training_steps = args.epochs * len(trainset) * steps_per_image
    total_steps = (training_steps+training_steps/args.sleep_every_n_images*args.rotations*steps_per_image*(args.sleep_cycles+args.rest_cycles))
    
    # Safely handle undefined hdf5_every / rate_every from the parent parser if missing
    if hasattr(args, 'hdf5_every') and args.hdf5_every:
        args.hdf5_n_snapshots = max(1, total_steps // args.hdf5_every)
    if hasattr(args, 'rate_every') and args.rate_every:
        args.rate_n_snapshots = max(1, total_steps // args.rate_every)

    with build_session(args) as session:
        train_mnist_unsupervised(
            session,
            args,
            epochs=args.epochs,
            steps_per_img=args.steps_per_img,
            warm_up_per_img=args.warm_up_per_img,
            seed=args.seed,
            log_every=args.log_every,
            snapshot_dir=args.snapshot_dir,
            snapshot_every=args.snapshot_every,
            snapshot_prefix=args.snapshot_prefix,
            freeze=False
        )