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


parser = argparse.ArgumentParser(description="Multi-seed gait cycle trajectory analysis.")
parser.add_argument("--logdir", type=str, required=True, help="Parent folder containing seed subfolders")
parser.add_argument("--title", type=str, required=True, help="Plot title (also used as Variant name in CSV)")
parser.add_argument("--output_csv", type=str, default="gait_metrics.csv", help="Filename for output CSV")
parser.add_argument("--benchmark", action="store_true", help="If plotting the benchmark policy")
parser.add_argument("--hind", action="store_true", help="Plot hind legs instead of front")


def normalize_to_gait_cycle(all_x, all_z_ground, all_contacts_bool, n_points=100):
    normalized_x = []
    normalized_z = []

    for env_idx in range(all_x.shape[0]):
        # Detect stance rising edges, using float conversion to be safe with boolean subtraction
        contact_float = all_contacts_bool[env_idx].astype(float)
        stance_rising = (contact_float - np.roll(contact_float, 1)) > 0.5
        stance_starts = np.where(stance_rising)[0]

        # Iterate through all detected complete cycles in this run
        for i in range(len(stance_starts) - 1):
            cycle_start = stance_starts[i]
            cycle_end = stance_starts[i+1]

            # Filter valid cycle length to avoid noise
            if (cycle_end - cycle_start > 20) and (cycle_end < all_x.shape[1]):
                cycle_x = all_x[env_idx, cycle_start:cycle_end]
                cycle_z = all_z_ground[env_idx, cycle_start:cycle_end]

                # Check for outliers (e.g. robot flying or falling through floor)
                if np.max(cycle_z) > 0.5 or np.max(cycle_z) < -0.1: continue 

                # Normalize time to 0 to 100 percent
                gait_phase = np.linspace(0, 100, n_points)
                old_phase = np.linspace(0, 100, len(cycle_x))

                x_interp = np.interp(gait_phase, old_phase, cycle_x)
                z_interp = np.interp(gait_phase, old_phase, cycle_z)

                normalized_x.append(x_interp)
                normalized_z.append(z_interp)

    if normalized_x:
        return np.array(normalized_x), np.array(normalized_z)

    return None, None

def load_seed_data(seed_dir):
    """Load one seed's npy files"""
    try:
        feet_positions = np.load(os.path.join(seed_dir, "feet_position.npy"))
        feet_contacts = np.load(os.path.join(seed_dir, "feet_contacts.npy"))
        hip_position_z = np.load(os.path.join(seed_dir, "hip_position_z.npy"))
        return feet_positions, feet_contacts, hip_position_z
    except FileNotFoundError as e:
        print(f"Skipping {seed_dir}: {e}")
        return None

if __name__ == "__main__":
    args = parser.parse_args()

    seed_dirs = glob.glob(os.path.join(args.logdir, "*"))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d) and all(f in os.listdir(d) for f in ["feet_position.npy", "feet_contacts.npy"])]
    seed_dirs.sort()

    print(f"Found {len(seed_dirs)} valid seed folders.")

    if not seed_dirs:
        print("No valid seed folders found!")
        exit()

    # Settings
    start_idx = 149 if args.benchmark else 299
    window_len = 250 if args.benchmark else 500
    time_window = slice(start_idx, start_idx + window_len)

    leg_pair = [2, 3] if args.hind else [0, 1]
    color = 'red' if args.hind else 'blue'

    # Store for global plot
    all_x_list = []
    all_z_list = []

    # Store for CSV
    seed_metrics_list = []

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        data = load_seed_data(seed_dir)
        if data is None:
            continue

        feet_positions, feet_contacts, hip_position_z = data

        all_hip_z = hip_position_z[:, time_window]
        all_x = feet_positions[:, leg_pair, 0, time_window].reshape(-1, time_window.stop - start_idx)
        all_z_body = feet_positions[:, leg_pair, 2, time_window].reshape(-1, time_window.stop - start_idx)
        all_contacts = feet_contacts[:, leg_pair, time_window].reshape(-1, time_window.stop - start_idx)

        all_contacts_bool = all_contacts > 0.5
        all_z_ground = all_z_body + np.repeat(all_hip_z, 2, axis=0)

        # Extract cycles for this seed
        norm_x, norm_z = normalize_to_gait_cycle(all_x, all_z_ground, all_contacts_bool)

        if norm_x is not None:
            # Per-Cycle Metric Calculation
            # Clearance (max height)
            cycle_clearances = np.max(norm_z, axis=1)

            # Step Length (peak-to-peak x-axis position)
            cycle_steplens = np.ptp(norm_x, axis=1)

            # Duration above 0.8 maximum height
            # Use broadcasting to compare each cycle against its own peak
            thresholds = 0.8 * cycle_clearances
            points_above = np.sum(norm_z > thresholds[:, None], axis=1)
            cycle_durations_ratio = points_above / 100.0

            # Aggregate Stats for this seed
            stats = {
                "Variant": args.title,
                "Seed": seed_name,
                "N_Cycles": len(norm_x),
                "Clearance_Mean": np.mean(cycle_clearances),
                "Clearance_Std": np.std(cycle_clearances),
                "StepLen_Mean": np.mean(cycle_steplens),
                "StepLen_Std": np.std(cycle_steplens),
                "TimeAbove80_Mean": np.mean(cycle_durations_ratio),
                "TimeAbove80_Std": np.std(cycle_durations_ratio),
            }
            seed_metrics_list.append(stats)
            
            # Collect for global plot
            all_x_list.append(norm_x)
            all_z_list.append(norm_z)

            print(f"Seed: {seed_name}, Cycles: {len(norm_x)}")

    if not all_x_list:
        print("No valid gait cycles across any seeds!")
        exit()

    # Save CSV
    df = pd.DataFrame(seed_metrics_list)
    cols = [
        "Variant",
        "Seed",
        "N_Cycles",
        "Clearance_Mean",
        "Clearance_Std",
        "StepLen_Mean",
        "StepLen_Std",
        "TimeAbove80_Mean",
        "TimeAbove80_Std",
    ]
    df = df[cols]
    
    file_exists = os.path.isfile(args.output_csv)
    df.to_csv(args.output_csv, mode='a', header=not file_exists, index=False)
    print(f"\nStats saved to {args.output_csv}")

    # Calculate and print pooled stats
    pooled_clearance_mean = np.mean(df["Clearance_Mean"])
    pooled_clearance_std = np.std(df["Clearance_Mean"])

    pooled_steplen_mean = np.mean(df["StepLen_Mean"])
    pooled_steplen_std = np.std(df["StepLen_Mean"])

    pooled_time80_mean = np.mean(df["TimeAbove80_Mean"])
    pooled_time80_std = np.std(df["TimeAbove80_Mean"])

    print(f"\nPooled results: {args.title} ({len(df)} seeds)")
    print(f"Clearance: {pooled_clearance_mean:.4f} ± {pooled_clearance_std:.4f} m")
    print(f"Step Length: {pooled_steplen_mean:.4f} ± {pooled_steplen_std:.4f} m")
    print(f"Time > 0.8h: {pooled_time80_mean:.4f} ± {pooled_time80_std:.4f}\n")

    # Plot
    all_x = np.concatenate(all_x_list)
    all_z = np.concatenate(all_z_list)
    
    mean_x_cycle = np.mean(all_x, axis=0)
    mean_z_cycle = np.mean(all_z, axis=0)
    std_z_cycle = np.std(all_z, axis=0)

    fig, ax = plt.subplots(figsize=(4, 4))

    gait_phase_pct = np.linspace(0, 100, 100)
    phase_colors = plt.cm.viridis(gait_phase_pct / 100)
    for i in range(0, 100, 5):
        ax.plot(mean_x_cycle[i:i+2], mean_z_cycle[i:i+2], color=phase_colors[i], linewidth=6, alpha=0.7)

    ax.plot(mean_x_cycle, mean_z_cycle, color=color, linewidth=4, label='Mean', alpha=0.9, zorder=4)
    ax.fill_between(mean_x_cycle, mean_z_cycle - std_z_cycle, mean_z_cycle + std_z_cycle, color=color, alpha=0.3, label=r'$\pm 1 \sigma z$', zorder=3)
    ax.scatter(mean_x_cycle, mean_z_cycle, c=gait_phase_pct, cmap='viridis', s=30, edgecolors='black', linewidth=0.3, zorder=5)

    peak_idx = np.argmax(mean_z_cycle)
    ax.scatter(mean_x_cycle[peak_idx], mean_z_cycle[peak_idx], c='orange', s=150, zorder=10, label='Peak', edgecolors='black', linewidth=1.5)

    ax.plot(mean_x_cycle, mean_z_cycle, 'k-', linewidth=1.5, alpha=0.6, zorder=2)

    ax.set_aspect('equal')
    ax.set_title(args.title)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$z$ (m)')
    ax.grid(True, alpha=0.3)

    start_m = [0.14, 0.0]
    plot_size_m = [0.15, 0.13]
    ax.set_xlim(start_m[0], start_m[0] + plot_size_m[0])
    ax.set_ylim(start_m[1], start_m[1] + plot_size_m[1])

    plt.tight_layout()
    plt.show()
