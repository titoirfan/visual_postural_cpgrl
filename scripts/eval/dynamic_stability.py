import os
import numpy as np
import argparse
import glob
import pandas as pd


parser = argparse.ArgumentParser(description="Disturbance metrics (Roll-Only + Excess over baseline).")
parser.add_argument("--logdir", type=str, default=None, help="Parent directory with seed subfolders.")
parser.add_argument("--push_time", type=float, default=5.0, help="Push application time (s).")
parser.add_argument("--window_duration", type=float, default=1.0, help="Transient window duration (s).")
parser.add_argument("--baseline_duration", type=float, default=2.0, help="Baseline gait duration (s).")


def compute_recovery_times(commands, velocities, death_status, times, push_time=5.0, push_axis=1, epsilon=0.15, settle_window=100):
    recovery_times = []
    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]

    for env_idx in survived_env_idx:
        command = commands[env_idx]
        velocity = velocities[env_idx]
        time = times[env_idx, 0, :]

        push_index = np.argmin(np.abs(time - push_time))
        vel_after_push = velocity[push_axis, push_index:]
        command_after_push = command[push_axis, push_index:]
        is_stable = np.abs(vel_after_push - command_after_push) <= epsilon

        recovery_idx = -1
        for t in range(len(is_stable) - settle_window):
            if np.all(is_stable[t:t + settle_window]):
                recovery_idx = t
                break

        if recovery_idx != -1:
            recovery_times.append(time[push_index + recovery_idx] - time[push_index])
        else:
            recovery_times.append(np.nan)

    return survived_env_idx, recovery_times

def compute_excess_disturbance_metrics(joint_positions, quaternions, death_status, times, push_time, window_duration_s, baseline_duration_s):
    excess_joint_ranges = []
    excess_rolls = []

    is_survived = np.logical_not(death_status[:, 0, :].any(axis=1))
    survived_env_idx = np.where(is_survived)[0]
    
    for env_idx in survived_env_idx:
        time = times[env_idx, 0, :]

        # Windows
        base_start_idx = max(0, np.argmin(np.abs(time - (push_time - baseline_duration_s))))
        base_end_idx = np.argmin(np.abs(time - push_time))

        trans_start_idx = base_end_idx
        trans_end_idx = np.argmin(np.abs(time - (push_time + window_duration_s)))

        # Joint excess
        base_joint = joint_positions[env_idx, :, base_start_idx:base_end_idx]
        baseline_joint_range = np.mean(np.ptp(base_joint, axis=1))

        trans_joint = joint_positions[env_idx, :, trans_start_idx:trans_end_idx]
        transient_joint_range = np.mean(np.ptp(trans_joint, axis=1))
        excess_joint = max(0, transient_joint_range - baseline_joint_range)

        # Roll excess
        base_quat = quaternions[env_idx, :, base_start_idx:base_end_idx]
        w_b, x_b, y_b, z_b = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
        roll_base = np.arctan2(2 * (w_b * x_b + y_b * z_b), 1 - 2 * (x_b * x_b + y_b * y_b))
        baseline_roll_rms = np.sqrt(np.mean(roll_base**2))

        trans_quat = quaternions[env_idx, :, trans_start_idx:trans_end_idx]
        w_t, x_t, y_t, z_t = trans_quat[0], trans_quat[1], trans_quat[2], trans_quat[3]
        roll_trans = np.arctan2(2 * (w_t * x_t + y_t * z_t), 1 - 2 * (x_t * x_t + y_t * y_t))
        transient_roll_peak = np.max(np.abs(roll_trans))
        excess_roll = max(0, transient_roll_peak - baseline_roll_rms)

        excess_joint_ranges.append(excess_joint)
        excess_rolls.append(excess_roll)

    return excess_joint_ranges, excess_rolls

def load_seed_data(seed_dir, eval_name, trim=1):
    logdir = os.path.join(seed_dir, eval_name)
    try:
        if not os.path.exists(os.path.join(logdir, "command.npy")):
            return None

        def load(name):
            return np.load(os.path.join(logdir, name))[..., :-trim]

        return {
            'commands': load("command.npy"),
            'death_status': load("death_status.npy"),
            'linear_vels': load("base_tracked_velocity.npy"),
            'positions': load("base_position.npy"),
            'quaternions': load("base_quaternion.npy"),
            'joint_positions': load("joint_position.npy")
        }
    except Exception as e:
        print(f"Skipping {seed_dir}: {e}")
        return None

def combined_variance_weighted(means, stds, counts):
    means = np.array(means)
    stds = np.array(stds)
    counts = np.array(counts)

    valid_mask = (counts > 0) & (~np.isnan(means))
    if not np.any(valid_mask):
        return np.nan, np.nan

    m = means[valid_mask]
    s = stds[valid_mask]
    n = counts[valid_mask]

    total_n = np.sum(n)
    combined_mean = np.sum(n * m) / total_n

    sum_sq = np.sum(n * (s**2 + m**2))
    combined_var = max(0.0, (sum_sq / total_n) - combined_mean**2)

    return combined_mean, np.sqrt(combined_var)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.logdir is None:
        print("Error: --logdir required.")
        exit(1)

    seed_dirs = glob.glob(os.path.join(args.logdir, "*"))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]

    print(f"Found {len(seed_dirs)} seeds.")
    print(f"Transient window length: {args.window_duration} s, Baseline window length: {args.baseline_duration} s")

    eval_names = ["stab_side_025", "stab_side_050", "stab_side_075", "stab_side_100", "stab_side_125", "stab_side_150"]
    results_summary = []

    per_seed_csv = []
    seed_names = []

    for eval_name in eval_names:
        print(f"\n{eval_name}...")

        stats_lists = {'success_rate': [], 't_rec': [], 'disp': [], 'joint_excess': [], 'roll_excess': []}

        for i, seed_dir in enumerate(seed_dirs):
            data = load_seed_data(seed_dir, eval_name)
            if data is None: 
                continue

            n_total = data['death_status'].shape[0]
            times = np.tile(np.linspace(0.0, 8.99, 899).reshape(1, 1, 899), (n_total, 1, 1))

            # Success rate
            is_alive = np.logical_not(data['death_status'][:, 0, :].any(axis=1))
            n_survived = np.sum(is_alive)
            success_rate = n_survived / n_total

            stats_lists['success_rate'].append((success_rate, 0.0, n_total))

            # Recovery time
            survived_idx, raw_t_rec = compute_recovery_times(data['commands'], data['linear_vels'], data['death_status'], times, push_time=args.push_time)

            if n_survived > 0:
                # Check if we have any valid recovery times
                valid_t_rec = np.array(raw_t_rec)
                valid_t_rec = valid_t_rec[~np.isnan(valid_t_rec)]

                if len(valid_t_rec) > 0:
                    t_rec_mean = np.mean(valid_t_rec)
                    t_rec_std = np.std(valid_t_rec)
                    # Count for weighting is the number of stabilized survivors
                    n_stabilized = len(valid_t_rec) 
                    stats_lists['t_rec'].append((t_rec_mean, t_rec_std, n_stabilized))
                else:
                    # Survived but never stabilized
                    stats_lists['t_rec'].append((np.nan, np.nan, 0))
            else:
                stats_lists['t_rec'].append((np.nan, np.nan, 0))

            # Excess metrics
            excess_joint, excess_roll = compute_excess_disturbance_metrics(data['joint_positions'], data['quaternions'], data['death_status'], times, args.push_time, args.window_duration, args.baseline_duration)
            if len(excess_joint) > 0:
                stats_lists['joint_excess'].append((np.mean(excess_joint), np.std(excess_joint), len(excess_joint)))
                stats_lists['roll_excess'].append((np.mean(excess_roll), np.std(excess_roll), len(excess_roll)))
            else:
                stats_lists['joint_excess'].append((np.nan, np.nan, 0))
                stats_lists['roll_excess'].append((np.nan, np.nan, 0))

            print(f"Seed: {os.path.basename(seed_dir)}, SR: {success_rate*100:.2f}%")

            seed_name = os.path.basename(seed_dir)
            seed_names.append(seed_name)
            
            per_seed_data = {
                'seed': seed_name,
                'eval': eval_name,
                'success_rate': success_rate,
                't_rec_mean': np.nanmean(valid_t_rec) if len(valid_t_rec)>0 else np.nan,
                't_rec_std': np.nanstd(valid_t_rec) if len(valid_t_rec)>0 else np.nan,
                't_rec_n': len(valid_t_rec),
                'joint_excess_mean': np.mean(excess_joint),
                'joint_excess_std': np.std(excess_joint),
                'joint_excess_n': n_total,
                'roll_excess_mean': np.mean(excess_roll),
                'roll_excess_std': np.std(excess_roll),
                'roll_excess_n': n_total
            }
            per_seed_csv.append(per_seed_data)

        # Save per-seed CSV
        df_seeds = pd.DataFrame(per_seed_csv)
        csv_path = os.path.join(args.logdir, "per_seed_disturbance_metrics.csv")
        df_seeds.to_csv(csv_path, index=False)
        print(f"Per-seed metrics saved to {csv_path}")

        # Aggregate metrics
        def aggregate(key):
            data = stats_lists[key]
            if not data:
                return np.nan, np.nan
            means = [x[0] for x in data]
            stds = [x[1] for x in data]
            counts = [x[2] for x in data]
            return combined_variance_weighted(means, stds, counts)

        # Calculate values immediately
        succ_mean, succ_std = aggregate('success_rate')
        trec_mean, trec_std = aggregate('t_rec')
        joint_mean, joint_std = aggregate('joint_excess')
        roll_mean, roll_std = aggregate('roll_excess')

        # Append flat dictionary to summary list
        results_summary.append({
            "Experiment": eval_name,
            "Success_Rate_Mean": succ_mean,
            "Success_Rate_Std": succ_std,
            "Recovery_Time_Mean": trec_mean,
            "Recovery_Time_Std": trec_std,
            "Joint_Excess_Mean": joint_mean,
            "Joint_Excess_Std": joint_std,
            "Roll_Excess_Mean": roll_mean,
            "Roll_Excess_Std": roll_std
        })

    # Save aggregated CSV
    df_summary = pd.DataFrame(results_summary)
    aggregated_csv_path = os.path.join(args.logdir, "disturbance_metrics_summary.csv")
    df_summary.to_csv(aggregated_csv_path, index=False)
    
    print(f"\nAggregated summary saved to: {aggregated_csv_path}")
