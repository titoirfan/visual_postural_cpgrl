import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob


plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["legend.loc"] = "lower left"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 11


parser = argparse.ArgumentParser(description="Visualize robot velocities across multiple seeds.")
parser.add_argument("--baseline", type=str, default=None, help="The parent directory containing baseline seed subfolders with npy files.")
parser.add_argument("--perceptive", type=str, default=None, help="The parent directory containing perceptive seed subfolders with npy files.")
parser.add_argument("--reflexive", type=str, default=None, help="The parent directory containing reflexive seed subfolders with npy files.")
parser.add_argument("--combined", type=str, default=None, help="The parent directory containing combined seed subfolders with npy files.")


def load_npy_data(parent_dir, filename, trim=200):
    """Load and concatenate base_tracked_velocity or similar from all seeds."""
    seed_dirs = glob.glob(os.path.join(parent_dir, "*"))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, filename))]

    all_data = []
    for seed_dir in seed_dirs:
        try:
            data = np.load(os.path.join(seed_dir, filename))[..., trim:-trim]
            all_data.append(data)
            print(f"Loaded {filename} from seed: {os.path.basename(seed_dir)} with shape {data.shape}")
        except FileNotFoundError:
            print(f"Skipping {seed_dir}: missing {filename}")
            continue

    if not all_data:
        raise FileNotFoundError(f"No valid seed directories with {filename} found in {parent_dir}")

    return np.concatenate(all_data, axis=0)

def plot_robot_velocities(commands, times, baseline, perceptive, reflexive, combined):
    time = times.squeeze(1)[0]

    mean_commands = np.mean(commands, axis=0)
    std_commands = np.std(commands, axis=0)
    mean_baseline = np.mean(baseline, axis=0)
    std_baseline = np.std(baseline, axis=0)
    mean_perceptive = np.mean(perceptive, axis=0)
    std_perceptive = np.std(perceptive, axis=0)
    mean_reflexive = np.mean(reflexive, axis=0)
    std_reflexive = np.std(reflexive, axis=0)
    mean_combined = np.mean(combined, axis=0)
    std_combined = np.std(combined, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(7, 4), sharex=False)

    tracking_data = [
        {'index': 0, 'ylabel': "$v_{b,x}$ (m/s)", 'time_range': (0, 10)},
        {'index': 1, 'ylabel': "$v_{b,y}$ (m/s)", 'time_range': (8, 16)},
        {'index': 2, 'ylabel': "$\omega_{b,z}$ (rad/s)", 'time_range': (14, 22)}
    ]

    colors = {
        'baseline': "#E69F00",
        'perceptive': "#56B4E9",
        'reflexive': "#009E73", 
        'combined': "#D55E00",
    }

    labels = {
        'commands': r"Command",
        'baseline': r"Base",
        'perceptive': r"Perceptive",
        'reflexive': r"Postural",
        'combined': r"Full",
    }

    for i, data in enumerate(tracking_data):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        index = data['index']
        ylabel = data['ylabel']
        time_start, time_end = data['time_range']

        start_index = np.argmin(np.abs(time - time_start))
        end_index = np.argmin(np.abs(time - time_end))

        t_slice = time[start_index:end_index]

        # Command
        ax.plot(t_slice, mean_commands[index, start_index:end_index], 
                label=labels['commands'], linewidth=2, color='black', zorder=1, linestyle='--')

        # Policies with mean ± std bands
        for policy, color in colors.items():
            mean_slice = locals()[f'mean_{policy}'][index, start_index:end_index]
            std_slice = locals()[f'std_{policy}'][index, start_index:end_index]
            
            # Fill first
            ax.fill_between(t_slice, mean_slice - std_slice, mean_slice + std_slice, 
                           color=color, alpha=0.25, zorder=3)
            # Mean line on top
            ax.plot(t_slice, mean_slice, label=labels[policy], 
                   color=color, linewidth=2.5, zorder=5)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'Time (s)')
        ax.grid(True, alpha=0.3, zorder=0)

    # Legend
    handles, labels_txt = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_txt, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
              ncols=5, fontsize=11, frameon=False, fancybox=False, shadow=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    required_args = ["baseline", "perceptive", "reflexive", "combined"]
    if any(getattr(args, arg) is None for arg in required_args):
        print("Error: All --baseline, --perceptive, --reflexive, --combined must be provided.")
        parser.print_help()
        exit(1)

    try:
        print("Loading npy files.")
        commands = load_npy_data(args.combined, "command.npy")
        times = np.tile(np.linspace(0.0, 23.99, 2399).reshape(1, 1, 2399), (1024, 1, 1))[..., 200:-200]

        baseline = load_npy_data(args.baseline, "base_tracked_velocity.npy")
        perceptive = load_npy_data(args.perceptive, "base_tracked_velocity.npy")
        reflexive = load_npy_data(args.reflexive, "base_tracked_velocity.npy")
        combined = load_npy_data(args.combined, "base_tracked_velocity.npy")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    plot_robot_velocities(commands, times, baseline, perceptive, reflexive, combined)
