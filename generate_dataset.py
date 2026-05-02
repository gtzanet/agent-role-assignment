#!/usr/bin/env python3
"""Generate Stage 1 dataset independently and save it under data/ with a timestamp."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from environment import CausalGraph, NodeType
from run_experiment import (
    REPO_ROOT,
    WORKFLOW_SIM_ROOT,
    _infer_causal_graph_for_algorithm,
    _load_node_metadata_from_config,
    _resolve_workflow_config_path,
)
from sim_utils import (
    build_dataset_row,
    build_runtime,
    load_simulator_config,
    run_single_sample,
    save_rows_to_csv,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_dataset_config() -> dict[str, object]:
    return {
        "config": "sim_config.yaml",
        "workflow_config": "workflow_causal_graph.yaml",
        "output_base_dir": "data",
        "dataset_id": None,
        "dataset_filename": "dataset.csv",
        "iterations": None,
        "min_samples": 10,
        "max_samples": 200,
        "convergence_threshold": 0.05,
        "convergence_window": 80,
        "perturb_threads": True,
        "perturb_prob": 0.50,
        "thread_min": 1,
        "thread_max": 4,
        "seed": None,
        "causal_threshold": 0.1,
        "causal_algorithm": "correlation",
    }


def _load_dataset_config(config_path: Path) -> argparse.Namespace:
    defaults = _default_dataset_config()
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at top level.")

    unknown = sorted(set(payload.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            f"Unknown keys in {config_path}: {unknown}. "
            f"Allowed keys: {sorted(defaults.keys())}"
        )

    merged = dict(defaults)
    merged.update(payload)

    causal_algorithms = {"notears", "correlation", "lingam"}
    if str(merged["causal_algorithm"]) not in causal_algorithms:
        raise ValueError(
            f"'causal_algorithm' must be one of {sorted(causal_algorithms)}."
        )

    return argparse.Namespace(**merged)


def _create_dataset_dir(base_dir_arg: str, dataset_id: str | None) -> Path:
    base_dir = Path(base_dir_arg)
    if not base_dir.is_absolute():
        base_dir = REPO_ROOT / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    run_name = dataset_id or _timestamp()
    dataset_dir = base_dir / run_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _persist_dataset_config(dataset_dir: Path, args: argparse.Namespace) -> Path:
    out_path = dataset_dir / "dataset_config.yaml"
    out_path.write_text(yaml.safe_dump(vars(args), sort_keys=False))
    return out_path


def _persist_simulator_config(dataset_dir: Path, sim_config: dict[str, object]) -> Path:
    out_path = dataset_dir / "sim_config.yaml"
    out_path.write_text(yaml.safe_dump(sim_config, sort_keys=False))
    return out_path


def _resolve_dataset_sim_config_path(config_name: str) -> Path:
    candidate = Path(config_name)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Simulator config file not found: {candidate}")
        return candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    workflow_candidate = WORKFLOW_SIM_ROOT / candidate
    if workflow_candidate.exists():
        return workflow_candidate

    raise FileNotFoundError(
        f"Simulator config file not found in repo root or workflow_simulator: {config_name}"
    )


def _graph_changed_fraction(prev_cg: CausalGraph, curr_cg: CausalGraph, nodes: list[str]) -> float:
    allowed = set(nodes)

    prev_edges = {(str(u), str(v)) for u, v in prev_cg.graph.edges() if u in allowed and v in allowed and u != v}
    curr_edges = {(str(u), str(v)) for u, v in curr_cg.graph.edges() if u in allowed and v in allowed and u != v}

    n = len(nodes)
    denom = max(n * (n - 1), 1)
    return float(len(prev_edges.symmetric_difference(curr_edges))) / float(denom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only Stage 1 (simulation + data collection) using dataset_config.yaml."
    )
    parser.add_argument(
        "--dataset-config",
        default="dataset_config.yaml",
        help="Path to dataset YAML config (default: dataset_config.yaml)",
    )
    cli_args = parser.parse_args()

    config_path = Path(cli_args.dataset_config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    args = _load_dataset_config(config_path)
    setattr(args, "dataset_config_source", str(config_path))
    return args


def main() -> None:
    args = parse_args()
    dataset_dir = _create_dataset_dir(args.output_base_dir, args.dataset_id)
    output_csv = dataset_dir / str(args.dataset_filename)
    persisted_config = _persist_dataset_config(dataset_dir, args)

    if str(WORKFLOW_SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKFLOW_SIM_ROOT))

    from simulator.application import Application  # pyright: ignore[reportMissingImports]
    from simulator.infrastructure import Node as SimNode  # pyright: ignore[reportMissingImports]
    from simulator.simulation import Simulation  # pyright: ignore[reportMissingImports]

    sim_config_path = _resolve_dataset_sim_config_path(args.config)
    config = load_simulator_config(sim_config_path)
    persisted_sim_config = _persist_simulator_config(dataset_dir, config)

    infra_cfg = config.get("infrastructure", {})
    sim_cfg = config.get("simulation", {})
    reward_cfg = config.get("reward", {})

    cpu_max = int(infra_cfg.get("cpu_max", 4))
    iterations = int(args.iterations or sim_cfg.get("iterations", 500))
    timeout = int(sim_cfg.get("timeout", 600000))
    eval_interval = float(sim_cfg.get("eval_interval", 10.0))
    latency_target = reward_cfg.get("e2e_lat_target", None)
    seed = int(args.seed if args.seed is not None else sim_cfg.get("seed", 42))

    np.random.seed(seed)

    runtime = build_runtime(
        config=config,
        simulation_classes=(Application, SimNode),
        perturb_threads=bool(args.perturb_threads),
        perturb_prob=float(args.perturb_prob),
        thread_min=int(args.thread_min),
        thread_max=int(args.thread_max),
    )

    n_nodes = len(runtime.nodes)
    n_services = runtime.n_services
    n_workflows = runtime.n_workflows

    rows: list[dict[str, float]] = []
    column_order: list[str] | None = None
    node_types: dict[str, NodeType] | None = None
    decision_space_sizes: dict[str, int] | None = None
    prev_graph: CausalGraph | None = None
    consecutive_stable = 0
    workflow_config_path = _resolve_workflow_config_path(args.workflow_config)

    print("=== Collect dataset (Simulator) ===")
    print(f"  Config: {sim_config_path}")
    print(f"  Dataset directory: {dataset_dir}")
    print(f"  Output: {output_csv}")
    print(f"  Saved config: {persisted_config}")
    print(f"  Saved simulator config: {persisted_sim_config}")
    print(f"  Nodes: {n_nodes}  Services: {n_services}  Workflows: {n_workflows}")
    print(f"  Iterations per run: {iterations}  eval_interval: {eval_interval}s")
    print(f"  Min/max samples: {args.min_samples} / {args.max_samples}")
    print(
        "  Convergence threshold: "
        f"{args.convergence_threshold} "
        f"(window: {args.convergence_window} consecutive, algo={args.causal_algorithm})"
    )

    for sample_idx in range(args.max_samples):
        run_single_sample(
            simulation_cls=Simulation,
            runtime=runtime,
            iterations=iterations,
            timeout=timeout,
            eval_interval=eval_interval,
            latency_target=latency_target,
        )

        if not runtime.collector.windows:
            print(f"  [{sample_idx + 1:>4}] No eval windows collected - skipping")
            continue

        row = build_dataset_row(runtime.collector, runtime.nodes, cpu_max, n_services, n_workflows)
        if column_order is None:
            column_order = list(row.keys())
            node_types, decision_space_sizes = _load_node_metadata_from_config(workflow_config_path, column_order)

        rows.append(row)
        save_rows_to_csv(rows, output_csv)

        status = f"  [{sample_idx + 1:>4}] {len(runtime.collector.windows)} windows -> {len(rows)} rows"

        if len(rows) < args.min_samples:
            print(status)
            continue

        if node_types is None or decision_space_sizes is None or column_order is None:
            raise RuntimeError("Node metadata was not initialized before convergence check.")

        curr_graph, _ = _infer_causal_graph_for_algorithm(args, output_csv, node_types, decision_space_sizes)
        n_edges = curr_graph.graph.number_of_edges()

        if prev_graph is None:
            print(f"{status}  |  edges={n_edges}  (first graph)")
            prev_graph = curr_graph
            continue

        changed_fraction = _graph_changed_fraction(prev_graph, curr_graph, column_order)
        stable = changed_fraction <= args.convergence_threshold
        consecutive_stable = consecutive_stable + 1 if stable else 0
        print(
            f"{status}  |  edges={n_edges}  "
            f"changed={changed_fraction:.4f}  "
            f"stable={consecutive_stable}/{args.convergence_window}"
        )
        prev_graph = curr_graph

        if consecutive_stable >= args.convergence_window:
            print(
                f"\nConverged after {len(rows)} samples "
                f"({args.convergence_window} consecutive stable comparisons)."
            )
            break

    if not output_csv.exists():
        raise FileNotFoundError(f"Expected dataset not found at: {output_csv}")

    print(f"[OK] Dataset generated: {output_csv}")


if __name__ == "__main__":
    main()
