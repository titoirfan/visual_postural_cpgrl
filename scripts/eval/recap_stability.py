import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 11


parser = argparse.ArgumentParser(description="Plot disturbance metrics with std shading (bottom legend).")
parser.add_argument("--baseline", type=str, required=True, help="Path to baseline metrics file.")
parser.add_argument("--perceptive", type=str, required=True, help="Path to perceptive metrics file.")
parser.add_argument("--reflexive", type=str, required=True, help="Path to reflexive metrics file.")
parser.add_argument("--combined", type=str, required=True, help="Path to combined metrics file.")


def parse_metrics_file(filepath):
    try:
        df = pd.read_csv(filepath)
        return {
            "success_rate": df["Success_Rate_Mean"].values,
            "success_std": df["Success_Rate_Std"].values,
            "recovery_time": df["Recovery_Time_Mean"].values,
            "recovery_std": df["Recovery_Time_Std"].values,
            "joint_excess": df["Joint_Excess_Mean"].values,
            "joint_std": df["Joint_Excess_Std"].values,
            "roll_excess": df["Roll_Excess_Mean"].values,
            "roll_std": df["Roll_Excess_Std"].values,
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        exit(1)

def compute_adaptive_ylim(all_data, metric_key, std_key, padding=0.05):
    """Adaptive ylim including std devs."""
    all_means = np.concatenate([d[metric_key] for d in all_data])
    all_stds = np.concatenate([d[std_key] for d in all_data])

    min_val = (all_means - all_stds).min()
    max_val = (all_means + all_stds).max()
    range_val = max_val - min_val if max_val > min_val else 1.0
    pad = range_val * padding
    return max(0.0, min_val - pad), max_val + pad

def plot_metrics_comparison(baseline_data, perceptive_data, reflexive_data, combined_data):
    all_data = [baseline_data, perceptive_data, reflexive_data, combined_data]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.0))

    metrics = [
        ("recovery_time", "recovery_std", "Recovery time (s)", axes[0]),
        ("joint_excess", "joint_std", "Joint motion range increase (rad)", axes[1]),
        ("roll_excess", "roll_std", "Peak body roll deviation (rad)", axes[2]),
    ]

    colors = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]
    labels = ["Base", "Perceptive", "Postural", "Full"]

    handles = []
    legend_labels = []

    ylim = [
        (0.0, 0.96),
        (0.0, 0.27),
        (0.0, 0.29),
    ]

    for m_idx, (mean_key, std_key, ylabel, ax) in enumerate(metrics):
        x_pos = np.arange(6)

        for i, data in enumerate(all_data):
            means = data[mean_key]
            stds = data[std_key]

            ax.fill_between(x_pos, means - stds, means + stds, color=colors[i], alpha=0.15)
            line, = ax.plot(x_pos, means, "o-", color=colors[i], linewidth=2.0, markersize=5)

            # Collect handles for legend
            if m_idx == 0:
                handles.append(line)
                legend_labels.append(labels[i])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["0.25", "0.5", "0.75", "1.0", "1.25", "1.5"])
        ax.set_xlabel("Push magnitude (m/s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        ax.set_ylim(ylim[m_idx])

    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncols=4, fontsize=11, frameon=False, fancybox=False, shadow=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading...")
    baseline_data = parse_metrics_file(args.baseline)
    perceptive_data = parse_metrics_file(args.perceptive)
    reflexive_data = parse_metrics_file(args.reflexive)
    combined_data = parse_metrics_file(args.combined)

    print("Plotting...")
    plot_metrics_comparison(baseline_data, perceptive_data, reflexive_data, combined_data)
