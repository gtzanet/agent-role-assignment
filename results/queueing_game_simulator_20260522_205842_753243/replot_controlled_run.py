#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd



def build_plot(run_dir):
    run_dir = Path(run_dir)
    data = pd.read_csv(run_dir / "controlled_eval_records.csv")
    data = data.sort_values(["service_idx", "t"]).reset_index(drop=True)

    service_ids = [int(x) for x in sorted(data["service_idx"].unique())]
    service_names = {
        int(row.service_idx): row.service_name
        for row in data[["service_idx", "service_name"]].drop_duplicates().itertuples(index=False)
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    cmap = plt.get_cmap("tab10")

    for color_index, service_id in enumerate(service_ids):
        group = data[data["service_idx"] == service_id]
        color = cmap(color_index % 10)
        moving_average = group["chosen_r"].rolling(10, min_periods=1).mean()
        axes[0].step(group["t"], group["chosen_r"], where="post", color=color, alpha=0.3, linewidth=1)
        axes[0].plot(
            group["t"],
            moving_average,
            color=color,
            linewidth=2,
            label=f"{service_names.get(service_id, f'Service {service_id}')} (MA-10)",
        )
        axes[1].plot(
            group["t"],
            group["e2e_latency_ms"],
            color=color,
            label=service_names.get(service_id, f"Service {service_id}"),
        )

    axes[0].set_xlabel("Simulation time (s)")
    axes[0].set_ylabel("Replicas")
    axes[0].set_title("Replica count — raw + 10-tick moving average")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Mean e2e latency (ms)")
    axes[1].set_title("Latency per eval window — controlled")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=".")
    parser.add_argument("--output", default="controlled_latency_and_replicas_reloaded.pdf")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    fig = build_plot(run_dir)
    output_path = run_dir / args.output
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")



if __name__ == "__main__":
    main()
