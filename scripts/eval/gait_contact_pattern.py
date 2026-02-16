import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import pandas as pd


plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["legend.fontsize"] = 14


parser = argparse.ArgumentParser(description="Quadruped gait contact pattern analysis")
parser.add_argument("--logdir", type=str, required=True, help="Parent folder containing seed subfolders")
parser.add_argument("--title", type=str, required=True, help="Variant name for CSV (e.g., Base, Perceptive, Postural, Full)")
parser.add_argument("--output_csv", type=str, default="gait_stats.csv", help="Filename for output CSV")
parser.add_argument("--benchmark", action="store_true", help="If plotting the benchmark policy")


def calculate_step_metrics(all_contacts, dt=0.01):
    """
    Calculate average step frequency (Hz) and duty factor per seed.
    Input: all_contacts [envs, 4, time]
    """
    all_freqs = []
    all_duty_factors = []

    # Iterate over all environments in this seed
    for env_idx in range(all_contacts.shape[0]):
        # Calculate duty factor across all legs and time - mean of the boolean contact array
        env_duty = np.mean(all_contacts[env_idx])
        all_duty_factors.append(env_duty)

        # Calculate frequency per leg
        for leg_idx in range(4):
            leg_contacts = all_contacts[env_idx, leg_idx, :]

            # Detect rising edges
            stance_rising = (leg_contacts > 0.5) & ~np.roll(leg_contacts > 0.5, 1)
            stance_starts = np.where(stance_rising)[0]

            if len(stance_starts) < 2:
                continue

            # Time difference between steps (stride period)
            stride_durations = np.diff(stance_starts) * dt

            # Filter noise - steps faster than 10 Hz or slower than 0.5 Hz
            valid_durations = stride_durations[(stride_durations > 0.1) & (stride_durations < 2.0)]

            if len(valid_durations) > 0:
                freqs = 1.0 / valid_durations
                all_freqs.extend(freqs)

    if not all_freqs:
        return np.nan, np.nan, np.nan, np.nan

    return (np.mean(all_freqs), np.std(all_freqs), np.mean(all_duty_factors), np.std(all_duty_factors))

def normalize_to_gait_cycle(all_contacts, n_points=100):
    """Extract normalized gait cycles from contact data"""
    normalized_contacts = []
    
    # Ensure all_contacts is float and proper shape [envs, legs, time]
    all_contacts = all_contacts.astype(np.float32)

    for env_idx in range(all_contacts.shape[0]):
        # Use leg 0 (front left) as the phase reference.
        ref_leg_contacts = all_contacts[env_idx, 0, :] 

        stance_mask = ref_leg_contacts > 0.5
        stance_rising = stance_mask & ~np.roll(stance_mask, 1)
        stance_starts = np.where(stance_rising)[0]

        for i in range(len(stance_starts) - 1):
            cycle_start = stance_starts[i]
            cycle_end = stance_starts[i+1]

            if (cycle_end - cycle_start > 20) and (cycle_end < all_contacts.shape[2]):
                cycle_contacts = all_contacts[env_idx, :, cycle_start:cycle_end]

                gait_phase = np.linspace(0, 100, n_points)
                norm_cycle = np.zeros((4, n_points))

                current_len = cycle_contacts.shape[1]
                old_times = np.linspace(0, 100, current_len)

                for leg in range(4):
                    norm_cycle[leg] = np.interp(gait_phase, old_times, cycle_contacts[leg])

                normalized_contacts.append(norm_cycle)

    if normalized_contacts:
        return np.array(normalized_contacts)
    return None

def load_seed_data(seed_dir):
    try:
        feet_contacts = np.load(os.path.join(seed_dir, "feet_contacts.npy"))
        if feet_contacts.dtype != np.float32:
            feet_contacts = feet_contacts.astype(np.float32)
        return feet_contacts
    except Exception as e:
        print(f"Skipping {seed_dir}: {e}")
        return None

if __name__ == "__main__":
    args = parser.parse_args()
    
    seed_dirs = glob.glob(os.path.join(args.logdir, "*"))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d) and 
                "feet_contacts.npy" in os.listdir(d)]
    seed_dirs.sort()

    print(f"Found {len(seed_dirs)} valid seed folders")

    if not seed_dirs:
        print("No valid seed folders found!")
        exit()

    # Settings
    start_idx = 149 if args.benchmark else 299
    window_len = 250 if args.benchmark else 500
    dt = 0.02 if args.benchmark else 0.01
    time_window = slice(start_idx, start_idx + window_len)

    all_norm_contacts = []
    seed_stats_list = []

    for seed_dir in seed_dirs:
        data = load_seed_data(seed_dir)
        if data is None:
            continue

        feet_contacts = data[:, :, time_window]  # [env, leg, time]
        freq_mean, freq_std, df_mean, df_std = calculate_step_metrics(feet_contacts, dt=dt)
        
        seed_name = os.path.basename(seed_dir)

        # Save to list
        seed_stats_list.append({
            "Variant": args.title,
            "Seed": seed_name,
            "Freq_Mean": freq_mean,
            "Freq_Std": freq_std,
            "DutyFactor_Mean": df_mean,
            "DutyFactor_Std": df_std
        })

        print(f"Seed: {seed_name}, Freq: {freq_mean:.4f} ± {freq_std:.4f} Hz, DF: {df_mean:.4f} ± {df_std:.4f}")

        # Process visualization data
        norm_contacts = normalize_to_gait_cycle(feet_contacts)
        if norm_contacts is not None and len(norm_contacts) > 0:
            all_norm_contacts.append(norm_contacts)

    if not seed_stats_list:
        print("No valid data found.")
        exit()

    # Save CSV
    df = pd.DataFrame(seed_stats_list)
    file_exists = os.path.isfile(args.output_csv)
    df.to_csv(args.output_csv, mode='a', header=not file_exists, index=False)
    print(f"\nStats saved to {args.output_csv}")

    # Pooled statistics
    all_freq_means = [s["Freq_Mean"] for s in seed_stats_list]
    all_df_means = [s["DutyFactor_Mean"] for s in seed_stats_list]

    print(f"\nPooled results: {args.title}")
    print(f"Frequency: {np.mean(all_freq_means):.4f} ± {np.std(all_freq_means):.4f} Hz")
    print(f"Duty Factor: {np.mean(all_df_means):.4f} ± {np.std(all_df_means):.4f}\n")

    if not all_norm_contacts:
        print("No valid gait cycles for plotting.")
        exit()

    # Plot contact probability heatmap
    all_contacts = np.concatenate(all_norm_contacts, axis=0)
    n_cycles = all_contacts.shape[0]
    mean_contacts = np.mean(all_contacts, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))

    im = ax.imshow(mean_contacts, cmap='viridis', aspect='auto', extent=[0, 100, 3.5, -0.5], vmin=0, vmax=1)
    ax.set_title(args.title)
    ax.set_xlabel('Gait cycle (\%)')
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(['FL', 'FR', 'HL', 'HR'])
    fig.colorbar(im, ax=ax, label='Contact probability')

    plt.tight_layout()
    plt.show()
