#!/usr/bin/env python3
"""Run the UML experiment sequence step-by-step (Stages 2-5).

Stage 1 (dataset generation) is handled exclusively by generate_dataset.py.
All runtime parameters are loaded from exp_config.yaml.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

from causality.causal_discovery import infer_causal_graph_notears, infer_causal_graph_correlation, infer_causal_graph_lingam
from causality.task_interference_analyzer import TaskInterferenceAnalyzer, TaskInterferenceGraph
from environment import CausalGraph, Node, NodeType
from sim_utils import (
    build_runtime,
    load_simulator_config,
    reset_app_threads,
)
def _infer_causal_graph_for_algorithm(
    args: argparse.Namespace,
    dataset_csv: Path,
    node_types: dict[str, NodeType],
    decision_space_sizes: dict[str, int],
) -> tuple[CausalGraph, str]:
    if args.causal_algorithm == "correlation":
        return (
            infer_causal_graph_correlation(
                str(dataset_csv),
                node_types=node_types,
                threshold=args.causal_threshold,
                decision_space_sizes=decision_space_sizes,
            ),
            "Correlation-tier",
        )
    if args.causal_algorithm == "lingam":
        return (
            infer_causal_graph_lingam(
                str(dataset_csv),
                node_types=node_types,
                threshold=args.causal_threshold,
                decision_space_sizes=decision_space_sizes,
            ),
            "DirectLiNGAM",
        )
    return (
        infer_causal_graph_notears(
            str(dataset_csv),
            node_types=node_types,
            threshold=args.causal_threshold,
            decision_space_sizes=decision_space_sizes,
        ),
        "CausalNex NOTEARS",
    )


REPO_ROOT = Path(__file__).resolve().parent
WORKFLOW_SIM_ROOT = REPO_ROOT / "workflow_simulator"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_exp_config() -> dict[str, object]:
    return {
        "experiments_dir": str(REPO_ROOT / "experiments"),
        "experiment_id": None,
        "step": 5,
        "dataset_dir": None,
        "workflow_config": "workflow_causal_graph.yaml",
        "iterations": None,
        "seed": None,
        "causal_threshold": 0.1,
        "causal_algorithm": "correlation",
        "delta_mode": "interventional",
        "delta_bins": 5,
        "delta_max_parents": 3,
        "delta_max_workers": None,
        "replica_action_space": 10,
        "thread_action_space": 4,
        "c_max": 100.0,
        "rho": 0.1,
        "partition_algorithm": "greedy_modularity",
        "n_agents": 2,
        "omega": None,
        "step5_train_episodes": None,
        "step5_train_iterations": None,
        "causal_analyzer_url": None,
    }


def _load_exp_config(config_path: Path) -> argparse.Namespace:
    defaults = _default_exp_config()
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")

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

    merged["step"] = int(merged["step"])
    if merged["step"] not in {2, 3, 4, 5}:
        raise ValueError("'step' must be one of [2, 3, 4, 5].")

    causal_algorithms = {"notears", "correlation", "lingam"}
    if str(merged["causal_algorithm"]) not in causal_algorithms:
        raise ValueError(
            f"'causal_algorithm' must be one of {sorted(causal_algorithms)}."
        )

    delta_modes = {"interventional", "reachability"}
    if str(merged["delta_mode"]) not in delta_modes:
        raise ValueError(f"'delta_mode' must be one of {sorted(delta_modes)}.")

    if merged.get("delta_max_workers") is not None:
        merged["delta_max_workers"] = int(merged["delta_max_workers"])
        if merged["delta_max_workers"] <= 0:
            raise ValueError("'delta_max_workers' must be greater than 0 when provided.")

    partition_algorithms = {"spectral", "greedy_modularity", "kernighan_lin"}
    if str(merged["partition_algorithm"]) not in partition_algorithms:
        raise ValueError(
            f"'partition_algorithm' must be one of {sorted(partition_algorithms)}."
        )

    if not merged.get("dataset_dir"):
        raise ValueError("'dataset_dir' is required in exp_config.yaml.")

    return argparse.Namespace(**merged)


def _resolve_dataset_dir(dataset_dir_arg: str) -> Path:
    dataset_dir = Path(dataset_dir_arg)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def _load_dataset_artifacts(dataset_dir: Path) -> tuple[Path, Path, Path]:
    dataset_config_path = dataset_dir / "dataset_config.yaml"
    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"dataset_config.yaml not found in dataset directory: {dataset_dir}"
        )

    dataset_config = yaml.safe_load(dataset_config_path.read_text()) or {}
    if not isinstance(dataset_config, dict):
        raise ValueError(f"{dataset_config_path} must contain a YAML mapping.")

    dataset_filename = str(dataset_config.get("dataset_filename", "dataset.csv"))
    dataset_csv = dataset_dir / dataset_filename
    if not dataset_csv.exists():
        raise FileNotFoundError(
            f"Dataset CSV declared in {dataset_config_path} not found: {dataset_csv}"
        )

    sim_config_path = dataset_dir / "sim_config.yaml"
    if not sim_config_path.exists():
        raise FileNotFoundError(
            f"sim_config.yaml not found in dataset directory: {dataset_dir}"
        )

    return dataset_csv, sim_config_path, dataset_config_path


def _to_serializable_config_dict(args: argparse.Namespace) -> dict[str, object]:
    payload = vars(args).copy()
    return payload


def _persist_experiment_config(exp_dir: Path, args: argparse.Namespace) -> Path:
    out_path = exp_dir / "exp_config.yaml"
    out_path.write_text(yaml.safe_dump(_to_serializable_config_dict(args), sort_keys=False))
    return out_path


def _strip_agent_config(config: dict[str, object]) -> dict[str, object]:
    sanitized = dict(config)

    control_cfg = dict(sanitized.get("control", {}) or {})
    control_cfg.pop("agent_control_assignments", None)
    control_cfg["agent_controls_replicas"] = False
    control_cfg["agent_controls_placement"] = False
    sanitized["control"] = control_cfg

    reward_cfg = dict(sanitized.get("reward", {}) or {})
    reward_cfg.pop("agents", None)
    sanitized["reward"] = reward_cfg

    return sanitized


def _persist_agent_allocation_snapshot(exp_dir: Path) -> Path:
    agent_yaml = exp_dir / "agent_assignments.yaml"
    if not agent_yaml.exists():
        raise FileNotFoundError(f"Agent allocation file not found: {agent_yaml}")

    snapshot_path = exp_dir / "agent_allocation.yaml"
    snapshot_path.write_text(agent_yaml.read_text())
    return snapshot_path


def _create_experiment_dir(base_dir: Path, experiment_id: str | None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    exp_name = experiment_id or _timestamp()
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _run_command(cmd: list[str], cwd: Path) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    import subprocess

    subprocess.run(cmd, cwd=str(cwd), check=True)


def _serialize_causal_graph(cg: CausalGraph, graph_json_path: Path) -> None:
    payload = {
        "nodes": [
            {
                "name": n.name,
                "node_type": n.node_type.value,
                "decision_space_size": int(n.decision_space_size),
            }
            for n in sorted(cg.nodes.values(), key=lambda x: x.name)
        ],
        "edges": sorted([[str(u), str(v)] for u, v in cg.graph.edges()]),
    }
    graph_json_path.write_text(json.dumps(payload, indent=2))


def _deserialize_causal_graph(graph_json_path: Path) -> CausalGraph:
    payload = json.loads(graph_json_path.read_text())
    cg = CausalGraph()

    node_type_map = {
        NodeType.INPUT.value: NodeType.INPUT,
        NodeType.INTERMEDIARY.value: NodeType.INTERMEDIARY,
        NodeType.KPI.value: NodeType.KPI,
    }

    for nd in payload.get("nodes", []):
        nt = node_type_map.get(nd.get("node_type"), NodeType.INTERMEDIARY)
        cg.add_node(
            Node(
                str(nd["name"]),
                nt,
                int(nd.get("decision_space_size", 1)),
            )
        )

    for edge in payload.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = str(edge[0]), str(edge[1])
        if src in cg.nodes and dst in cg.nodes:
            cg.add_edge(src, dst)

    return cg


def _resolve_workflow_config_path(config_name: str) -> Path:
    config_path = Path(config_name)
    if config_path.is_absolute():
        return config_path
    return REPO_ROOT / config_path


def _load_node_metadata_from_config(config_path: Path, columns: list[str]) -> tuple[dict[str, NodeType], dict[str, int]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text()) or {}
    causal_graph_config = config.get("causal_graph", {})
    node_types_config = causal_graph_config.get("node_types", {})
    decision_space_config = causal_graph_config.get("decision_space_sizes", {})

    if not isinstance(node_types_config, dict) or not node_types_config:
        raise ValueError(f"{config_path} must define causal_graph.node_types")

    node_types: dict[str, NodeType] = {}
    decision_space_sizes: dict[str, int] = {}

    for name in columns:
        if name not in node_types_config:
            raise ValueError(f"Missing node type for column '{name}' in {config_path}")

        raw_node_type = str(node_types_config[name]).strip().lower()
        try:
            node_types[name] = NodeType(raw_node_type)
        except ValueError as exc:
            raise ValueError(
                f"Invalid node type '{raw_node_type}' for column '{name}' in {config_path}. "
                "Expected one of: input, intermediary, kpi."
            ) from exc

        decision_space_sizes[name] = int(decision_space_config.get(name, 1))

    return node_types, decision_space_sizes


def _load_variable_groups_from_config(config_path: Path, columns: list[str]) -> dict[str, list[str]]:
    node_types, _ = _load_node_metadata_from_config(config_path, columns)
    variable_groups: dict[str, list[str]] = {
        "inputs": [],
        "intermediates": [],
        "outputs": [],
    }

    for name in columns:
        node_type = node_types[name]
        if node_type == NodeType.INPUT:
            variable_groups["inputs"].append(name)
        elif node_type == NodeType.INTERMEDIARY:
            variable_groups["intermediates"].append(name)
        elif node_type == NodeType.KPI:
            variable_groups["outputs"].append(name)

    return variable_groups


def _save_causal_graph_plot(cg: CausalGraph, output_png: Path) -> None:
    """Save causal graph plot with node colors by semantic type."""
    g = cg.graph.copy()
    if g.number_of_nodes() == 0:
        return

    color_by_type = {
        NodeType.INPUT: "#4C78A8",
        NodeType.INTERMEDIARY: "#72B7B2",
        NodeType.KPI: "#E45756",
    }

    groups: dict[NodeType, list[str]] = {
        NodeType.INPUT: [],
        NodeType.INTERMEDIARY: [],
        NodeType.KPI: [],
    }
    for name, node in sorted(cg.nodes.items(), key=lambda item: item[0]):
        if name in g:
            groups[node.node_type].append(name)

    x_map = {
        NodeType.INPUT: 0.0,
        NodeType.INTERMEDIARY: 0.5,
        NodeType.KPI: 1.0,
    }

    pos: dict[str, tuple[float, float]] = {}
    for node_type, members in groups.items():
        x = x_map[node_type]
        n = len(members)
        for i, name in enumerate(members):
            y = 1.0 - i / max(n - 1, 1)
            pos[name] = (x, y)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")

    for node_type, members in groups.items():
        if not members:
            continue
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=members,
            ax=ax,
            node_size=1700,
            node_color=color_by_type[node_type],
            edgecolors="#2b2b2b",
            linewidths=1.3,
            alpha=0.95,
        )

    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8.5, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.7,
        edge_color="#355070",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=24,
        min_target_margin=24,
        alpha=0.85,
    )

    ax.text(0.0, 1.06, "Input", transform=ax.transData, ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(0.5, 1.06, "Intermediary", transform=ax.transData, ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(1.0, 1.06, "KPI", transform=ax.transData, ha="center", va="bottom", fontsize=10, color="#555555")

    legend_handles = [
        mpatches.Patch(color=color_by_type[NodeType.INPUT], label="Input"),
        mpatches.Patch(color=color_by_type[NodeType.INTERMEDIARY], label="Intermediary"),
        mpatches.Patch(color=color_by_type[NodeType.KPI], label="KPI"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9, title="Node Type", title_fontsize=9)

    ax.set_title(f"Causal Graph by Node Type ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)", fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_causal_graph(args: argparse.Namespace, exp_dir: Path, dataset_csv: Path) -> Path:
    print("\n=== Step 2/5: Build causal graph (CausalDiscovery) ===")
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset is required for Step 2: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    workflow_config_path = _resolve_workflow_config_path(args.workflow_config)
    node_types, decision_space_sizes = _load_node_metadata_from_config(workflow_config_path, list(df.columns))

    cg, algo_label = _infer_causal_graph_for_algorithm(args, dataset_csv, node_types, decision_space_sizes)

    edges = sorted((str(u), str(v)) for u, v in cg.graph.edges())
    nodes = sorted(cg.nodes.keys())
    inputs = sorted(n.name for n in cg.get_inputs())
    kpis = sorted(n.name for n in cg.get_kpis())

    edge_csv = exp_dir / "causal_graph_edges.csv"
    edge_csv.write_text("source,target\n" + "\n".join(f"{u},{v}" for u, v in edges) + ("\n" if edges else ""))

    summary_json = exp_dir / "causal_graph_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "dataset": str(dataset_csv),
                "threshold": args.causal_threshold,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "inputs": inputs,
                "kpis": kpis,
                "nodes": nodes,
            },
            indent=2,
        )
    )

    graph_json = exp_dir / "causal_graph.json"
    _serialize_causal_graph(cg, graph_json)

    graph_png = exp_dir / "causal_graph_plot.png"
    _save_causal_graph_plot(cg, graph_png)

    print(f"[OK] Causal graph built with {algo_label}: {len(nodes)} nodes, {len(edges)} edges")
    print(f"[OK] Saved edges: {edge_csv}")
    print(f"[OK] Saved summary: {summary_json}")
    print(f"[OK] Saved graph artifact: {graph_json}")
    print(f"[OK] Saved graph plot: {graph_png}")

    return edge_csv


def build_tig(args: argparse.Namespace, exp_dir: Path, dataset_csv: Path) -> tuple[Path, TaskInterferenceAnalyzer, TaskInterferenceGraph]:
    print("\n=== Step 3/5: Compute TIG (TIG) ===")
    step3_start = time.perf_counter()
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset is required for Step 3: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    print(f"[Step 3] Loaded dataset: {dataset_csv} ({len(df)} rows, {len(df.columns)} columns)")

    graph_json = exp_dir / "causal_graph.json"
    if not graph_json.exists():
        raise FileNotFoundError(
            f"Causal graph artifact is required for Step 3: {graph_json}. "
            "Run Step 2 first to generate it."
        )
    cg = _deserialize_causal_graph(graph_json)
    print(f"[OK] Loaded causal graph artifact: {graph_json}")

    workflow_config_path = _resolve_workflow_config_path(args.workflow_config)
    variable_groups = _load_variable_groups_from_config(workflow_config_path, list(df.columns))
    analyzer = TaskInterferenceAnalyzer(cg, variable_groups, causal_analyzer_url=args.causal_analyzer_url)

    omega = None
    if args.omega:
        omega = np.array([float(x.strip()) for x in args.omega.split(",") if x.strip()], dtype=float)

    tig = analyzer.generate_tig(
        df,
        output_dir=exp_dir,
        dataset_path=str(dataset_csv),
        causal_threshold=args.causal_threshold,
        delta_mode=args.delta_mode,
        bins=args.delta_bins,
        max_parents=args.delta_max_parents,
        max_workers=args.delta_max_workers,
        omega=omega,
        c_max=args.c_max,
        rho=args.rho,
        replica_action_space=args.replica_action_space,
        thread_action_space=args.thread_action_space,
        save_plot=True,
    )
    artifacts = analyzer.persist_tig(tig, output_dir=exp_dir, save_plot=True)

    tasks = tig.tasks
    kpis = tig.kpis
    w_csv = artifacts["w_csv"]
    delta_csv = artifacts["delta_csv"]
    edges_csv = artifacts["edges_csv"]
    summary_json = artifacts["summary_json"]
    tig_png = artifacts["plot_png"]

    print(f"[Step 3] Identified {len(tasks)} tasks and {len(kpis)} KPIs")
    print(f"[Step 3] Delta matrix computed in {time.perf_counter() - step3_start:.2f}s")
    print(f"[OK] TIG computed: {len(tasks)} tasks, {len(kpis)} KPIs")
    print(f"[OK] Saved delta matrix: {delta_csv}")
    print(f"[OK] Saved W matrix: {w_csv}")
    print(f"[OK] Saved TIG edges: {edges_csv}")
    print(f"[OK] Saved TIG summary: {summary_json}")
    print(f"[OK] Saved TIG plot: {tig_png}")

    return w_csv, analyzer, tig


def _save_tig_plot(tasks: list[str], w: np.ndarray, output_png: Path) -> None:
    """Save a TIG visualization with heatmap + weighted graph."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: weight heatmap
    im = axes[0].imshow(w, cmap="viridis")
    axes[0].set_title("TIG Weight Matrix")
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels(tasks, rotation=45, ha="right")
    axes[0].set_yticks(range(len(tasks)))
    axes[0].set_yticklabels(tasks)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Right panel: weighted undirected graph
    g = nx.Graph()
    g.add_nodes_from(tasks)
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if w[i, j] > 0:
                g.add_edge(tasks[i], tasks[j], weight=float(w[i, j]))

    pos = nx.spring_layout(g, seed=42)
    edge_widths = [1.0 + 5.0 * g[u][v]["weight"] for u, v in g.edges()] if g.number_of_edges() else []
    nx.draw_networkx_nodes(g, pos, ax=axes[1], node_size=1200, node_color="#89C2D9")
    nx.draw_networkx_labels(g, pos, ax=axes[1], font_size=9)
    if g.number_of_edges() > 0:
        nx.draw_networkx_edges(g, pos, ax=axes[1], width=edge_widths, alpha=0.8, edge_color="#1D3557")
    axes[1].set_title("TIG Graph View")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

def get_partition(
    args: argparse.Namespace,
    exp_dir: Path,
    analyzer: TaskInterferenceAnalyzer | None = None,
    tig: TaskInterferenceGraph | None = None,
) -> tuple[Path, Path]:
    print("\n=== Step 4/5: Partition tasks (Partitioner) ===")

    w_csv = exp_dir / "tig_W.csv"
    if not w_csv.exists():
        raise FileNotFoundError(f"TIG matrix not found. Run step 3 first: {w_csv}")

    w_df = pd.read_csv(w_csv, index_col=0)
    tasks = list(w_df.index)
    w = w_df.values.astype(float)

    if analyzer is None:
        graph_json = exp_dir / "causal_graph.json"
        if not graph_json.exists():
            raise FileNotFoundError(
                f"Causal graph artifact is required for Step 4: {graph_json}. Run Step 2 first to generate it."
            )
        cg = _deserialize_causal_graph(graph_json)

        workflow_config_path = _resolve_workflow_config_path(args.workflow_config)
        variable_groups = _load_variable_groups_from_config(workflow_config_path, tasks)
        analyzer = TaskInterferenceAnalyzer(cg, variable_groups, causal_analyzer_url=args.causal_analyzer_url)

    if len(tasks) == 0:
        raise RuntimeError("TIG matrix has no tasks to partition.")

    result = analyzer.partition_tig(
        tig=tig,
        w=w,
        tasks=tasks,
        n_agents=args.n_agents,
        algorithm=args.partition_algorithm,
        seed=args.seed,
        output_dir=exp_dir,
    )

    partitions = result["partitions"]
    partition_json = result["partition_json"]
    partition_csv = result["partition_csv"]
    summary_json = result["summary_json"]
    agent_yaml = result["agent_yaml"]

    print(f"[OK] Partitioned {len(tasks)} tasks into {len(partitions)} communities")
    print(f"[OK] Saved partitions: {partition_json}")
    print(f"[OK] Saved assignments: {partition_csv}")
    print(f"[OK] Saved summary: {summary_json}")
    print(f"[OK] Saved agent allocation: {agent_yaml}")

    return partition_json, agent_yaml


def _parse_service_id(service_token: str) -> int:
    token = str(service_token).strip().lower()
    if not token.startswith("s"):
        raise ValueError(f"Invalid service token '{service_token}'. Expected format s<id>.")
    return int(token[1:])


def _load_service_agent_mapping(agent_yaml: Path) -> dict[int, str]:
    if not agent_yaml.exists():
        raise FileNotFoundError(f"Agent assignment file not found: {agent_yaml}")

    payload = yaml.safe_load(agent_yaml.read_text()) or {}
    control = payload.get("control", payload)
    assignments = control.get("agent_control_assignments", {})
    if not isinstance(assignments, dict):
        raise ValueError(f"Invalid agent assignments in {agent_yaml}. Expected a mapping.")

    service_to_agent: dict[int, str] = {}
    for key, agent_name in assignments.items():
        parts = str(key).split(".", 1)
        if len(parts) != 2:
            continue
        service_token, control_name = parts
        if str(control_name).strip().lower() not in {"scaling", "replicas", "cpu"}:
            continue
        sid = _parse_service_id(service_token)
        service_to_agent[sid] = str(agent_name)

    if not service_to_agent:
        raise ValueError(f"No service scaling assignments found in {agent_yaml}")

    return service_to_agent


class _Step5KPIRecorder:
    def __init__(self, nodes: list[object], cpu_max: int, n_workflows: int):
        self.nodes = nodes
        self.cpu_max = max(float(cpu_max), 1e-12)
        self.n_workflows = int(n_workflows)
        self.rows: list[dict[str, float | int]] = []
        self.eval_step = 0
        self.sim_time = 0.0
        self._last_eval_payload_id: int | None = None
        self._cpu_cursors = {int(node.id): 0 for node in nodes}

    def record_eval(self, accumulated: dict) -> None:
        payload_id = id(accumulated)
        if payload_id == self._last_eval_payload_id:
            return
        self._last_eval_payload_id = payload_id

        self.eval_step += 1
        elapsed = float(accumulated.get("elapsed", 0.0))
        self.sim_time += elapsed

        workflows = accumulated.get("workflows", {})
        violation_rates = list(workflows.get("violation_rates", []))
        totals = list(workflows.get("total", []))

        row: dict[str, float | int] = {
            "step": self.eval_step,
            "elapsed": elapsed,
            "sim_time": self.sim_time,
        }

        total_violations = 0
        for wid in range(self.n_workflows):
            total = int(totals[wid]) if wid < len(totals) else 0
            rate = float(violation_rates[wid]) if wid < len(violation_rates) else 0.0
            violations = int(round(rate * total))

            row[f"wf{wid}_total_traces"] = total
            row[f"wf{wid}_violation_rate"] = rate
            row[f"wf{wid}_violations"] = violations
            total_violations += violations

        row["total_violations"] = total_violations

        for node in self.nodes:
            nid = int(node.id)
            start = self._cpu_cursors[nid]
            end = len(node.cpu_metric)
            samples = np.array(node.cpu_metric[start:end], dtype=float)
            self._cpu_cursors[nid] = end

            avg_threads = float(np.mean(samples)) if samples.size > 0 else 0.0
            avg_usage_pct = min(100.0, max(0.0, 100.0 * avg_threads / self.cpu_max))
            row[f"node{nid}_avg_cpu_threads"] = avg_threads
            row[f"node{nid}_avg_cpu_usage_pct"] = avg_usage_pct

        self.rows.append(row)


class _Step5ThreadControlDispatcher:
    """Dispatch service eval callbacks to shared per-agent thread controllers."""

    def __init__(
        self,
        service_to_agent: dict[int, str],
        cpu_max: int,
        recorder: _Step5KPIRecorder | None,
        epsilon: float,
        alpha: float,
        beta: float,
    ):
        service_agent_module = importlib.import_module("agents.agents")
        ServiceAgent = getattr(service_agent_module, "ServiceAgent")

        self.recorder = recorder
        self.service_to_agent = {int(service_id): str(agent_name) for service_id, agent_name in service_to_agent.items()}
        self.cpu_max = max(int(cpu_max), 1)
        self.qmax = max(8, self.cpu_max)
        self.agents_by_name: dict[str, object] = {}

        for agent_name in sorted(set(self.service_to_agent.values())):
            self.agents_by_name[agent_name] = ServiceAgent(
                epsilon=float(epsilon),
                alpha=float(alpha),
                beta=float(beta),
                actions=range(1, self.cpu_max + 1),
                qmax=self.qmax,
                goal_cpu=float(self.cpu_max),
                dimsMax=[self.qmax, self.cpu_max],
            )

    def on_eval(self, service_idx, service, accumulated_metrics, instant_metrics):
        if self.recorder is not None:
            self.recorder.record_eval(accumulated_metrics)
        agent_name = self.service_to_agent.get(int(service.id))
        if agent_name is None:
            return None

        agent = self.agents_by_name[agent_name]
        return agent.on_eval(service_idx, service, accumulated_metrics, instant_metrics)

    def reset_environment(self):
        for agent in self.agents_by_name.values():
            agent.reset_environment()

    def freeze_for_evaluation(self):
        for agent in self.agents_by_name.values():
            agent.epsilon = 0.0
            agent.beta = 0.0
            agent.vf.alpha = 0.0


def _run_step5_episode(
    simulation_cls: type,
    runtime: object,
    dispatcher: _Step5ThreadControlDispatcher,
    iterations: int,
    timeout: int,
    eval_interval: float,
    latency_target: float | None,
) -> None:
    mapped_agents: dict[int, _Step5ThreadControlDispatcher] = {
        service.id: dispatcher for service in runtime.app.services
    }

    reset_app_threads(runtime.app, thread_count=1)
    dispatcher.reset_environment()

    sim = simulation_cls(
        apps=[runtime.app],
        units=[],
        iterations=iterations,
        timeout=timeout,
        eval_interval=eval_interval,
        latency_target=latency_target,
    )
    sim.run(agents=mapped_agents)


def _build_baseline_mappings(runtime: object) -> dict[str, dict[int, str]]:
    app = runtime.app

    all_tasks = {service.id: "agent0" for service in app.services}
    per_task = {service.id: f"agent{service.id}" for service in app.services}

    service_to_workflows: dict[int, list[int]] = {int(service.id): [] for service in app.services}
    for workflow in app.workflows:
        for task_id in workflow.nodes:
            service_id = int(app.task_graph.nodes[task_id]["subset"])
            service_to_workflows.setdefault(service_id, []).append(int(workflow.id))

    per_workflow: dict[int, str] = {}
    workflow_to_agent: dict[int, str] = {}
    for workflow in sorted(app.workflows, key=lambda wf: int(wf.id)):
        workflow_to_agent[int(workflow.id)] = f"agent{int(workflow.id)}"

    for service_id in sorted(service_to_workflows):
        workflow_ids = sorted(set(service_to_workflows[service_id]))
        if workflow_ids:
            per_workflow[service_id] = workflow_to_agent[workflow_ids[0]]
        else:
            per_workflow[service_id] = "agent0"

    per_node: dict[int, str] = {}
    for service in app.services:
        per_node[int(service.id)] = f"agent{int(service.node.id)}"

    return {
        "selected": {},
        "one_agent_all_tasks": all_tasks,
        "one_agent_per_task": per_task,
        "one_agent_per_workflow": per_workflow,
        "one_agent_per_node": per_node,
    }


def _build_service_scopes(runtime: object) -> tuple[dict[int, list[int]], dict[int, int]]:
    app = runtime.app

    service_to_workflows: dict[int, list[int]] = {int(service.id): [] for service in app.services}
    for workflow in app.workflows:
        for task_id in workflow.nodes:
            service_id = int(app.task_graph.nodes[task_id]["subset"])
            service_to_workflows.setdefault(service_id, []).append(int(workflow.id))

    service_to_node = {int(service.id): int(service.node.id) for service in app.services}
    return service_to_workflows, service_to_node


def _build_reward_agents_config(
    service_to_agent: dict[int, str],
    service_to_workflows: dict[int, list[int]],
    service_to_node: dict[int, int],
    n_workflows: int,
    n_nodes: int,
) -> dict[str, dict[str, dict[int, float]]]:
    workflows_all = list(range(max(int(n_workflows), 0)))
    nodes_all = list(range(max(int(n_nodes), 0)))

    agent_to_services: dict[str, list[int]] = {}
    for service_id, agent_name in service_to_agent.items():
        agent_to_services.setdefault(str(agent_name), []).append(int(service_id))

    reward_agents: dict[str, dict[str, dict[int, float]]] = {}
    for agent_name, service_ids in sorted(agent_to_services.items()):
        workflow_scope = sorted(
            {
                int(wid)
                for sid in service_ids
                for wid in service_to_workflows.get(int(sid), [])
            }
        )
        node_scope = sorted({int(service_to_node.get(int(sid), -1)) for sid in service_ids if int(sid) in service_to_node})

        if not workflow_scope:
            workflow_scope = workflows_all
        if not node_scope:
            node_scope = nodes_all

        wf_weight = 1.0 / float(max(len(workflow_scope), 1))
        node_weight = 1.0 / float(max(len(node_scope), 1))

        reward_agents[agent_name] = {
            "workflows": {int(wid): float(wf_weight) for wid in workflow_scope},
            "nodes": {int(nid): float(node_weight) for nid in node_scope},
        }

    return reward_agents


def _write_scenario_sim_config(
    exp_dir: Path,
    scenario_name: str,
    base_config: dict,
    service_to_agent: dict[int, str],
    service_to_workflows: dict[int, list[int]],
    service_to_node: dict[int, int],
) -> Path:
    config_payload = dict(base_config)
    control_cfg = dict(config_payload.get("control", {}))
    reward_cfg = dict(config_payload.get("reward", {}))

    control_cfg["agent_controls_replicas"] = bool(control_cfg.get("agent_controls_replicas", True))
    control_cfg["agent_controls_placement"] = bool(control_cfg.get("agent_controls_placement", False))
    control_cfg["agent_control_assignments"] = {
        f"s{int(service_id)}.scaling": str(agent_name)
        for service_id, agent_name in sorted(service_to_agent.items())
    }

    topology_cfg = config_payload.get("topology", {})
    infra_cfg = config_payload.get("infrastructure", {})
    n_workflows = len(topology_cfg.get("workflows", []))
    n_nodes = int(infra_cfg.get("n_nodes", 0))
    reward_cfg["agents"] = _build_reward_agents_config(
        service_to_agent=service_to_agent,
        service_to_workflows=service_to_workflows,
        service_to_node=service_to_node,
        n_workflows=n_workflows,
        n_nodes=n_nodes,
    )

    config_payload["control"] = control_cfg
    config_payload["reward"] = reward_cfg

    scenario_config_path = exp_dir / f"evaluation_{scenario_name}_config.yaml"
    scenario_config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
    return scenario_config_path


def _run_step_5_scenario(
    args: argparse.Namespace,
    exp_dir: Path,
    scenario_name: str,
    service_to_agent: dict[int, str],
    sim_config_path: Path,
    cpu_max: int,
    iterations: int,
    timeout: int,
    eval_interval: float,
    latency_target: float | None,
    seed: int,
) -> dict[str, object]:
    if str(WORKFLOW_SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKFLOW_SIM_ROOT))

    from simulator.application import Application  # pyright: ignore[reportMissingImports]
    from simulator.infrastructure import Node as SimNode  # pyright: ignore[reportMissingImports]
    from simulator.simulation import Simulation  # pyright: ignore[reportMissingImports]

    config = load_simulator_config(sim_config_path)
    np.random.seed(seed)

    rl_cfg = config.get("rl", {})
    train_cfg = config.get("training", {})
    train_episodes = int(
        args.step5_train_episodes
        if args.step5_train_episodes is not None
        else train_cfg.get("n_episodes", 100)
    )
    train_iterations = int(
        args.step5_train_iterations
        if args.step5_train_iterations is not None
        else iterations
    )
    epsilon = float(rl_cfg.get("epsilon", 0.30))
    alpha = float(rl_cfg.get("alpha", 0.01))
    beta = float(rl_cfg.get("beta", 0.01))

    runtime = build_runtime(
        config=config,
        simulation_classes=(Application, SimNode),
        perturb_threads=False,
        perturb_prob=0.0,
        thread_min=1,
        thread_max=max(1, cpu_max),
    )

    dispatcher = _Step5ThreadControlDispatcher(
        service_to_agent=service_to_agent,
        cpu_max=cpu_max,
        recorder=None,
        epsilon=epsilon,
        alpha=alpha,
        beta=beta,
    )

    training_rows: list[dict[str, float | int]] = []
    for ep in range(train_episodes):
        ep_recorder = _Step5KPIRecorder(runtime.nodes, cpu_max=cpu_max, n_workflows=runtime.n_workflows)
        dispatcher.recorder = ep_recorder
        _run_step5_episode(
            simulation_cls=Simulation,
            runtime=runtime,
            dispatcher=dispatcher,
            iterations=train_iterations,
            timeout=timeout,
            eval_interval=eval_interval,
            latency_target=latency_target,
        )

        ep_df = pd.DataFrame(ep_recorder.rows)
        training_rows.append(
            {
                "episode": ep + 1,
                "total_violations": int(ep_df["total_violations"].sum()) if not ep_df.empty else 0,
                "mean_step_total_violations": float(ep_df["total_violations"].mean()) if not ep_df.empty else 0.0,
                "n_steps": int(len(ep_df)),
            }
        )

    training_df = pd.DataFrame(training_rows)
    training_csv = exp_dir / f"training_{scenario_name}_episodes.csv"
    training_df.to_csv(training_csv, index=False)

    dispatcher.freeze_for_evaluation()
    eval_recorder = _Step5KPIRecorder(runtime.nodes, cpu_max=cpu_max, n_workflows=runtime.n_workflows)
    dispatcher.recorder = eval_recorder
    _run_step5_episode(
        simulation_cls=Simulation,
        runtime=runtime,
        dispatcher=dispatcher,
        iterations=iterations,
        timeout=timeout,
        eval_interval=eval_interval,
        latency_target=latency_target,
    )

    detail_df = pd.DataFrame(eval_recorder.rows)
    if detail_df.empty:
        raise RuntimeError(f"Scenario '{scenario_name}' produced no evaluation rows.")

    detail_csv = exp_dir / f"evaluation_{scenario_name}_step_kpis.csv"
    detail_df.to_csv(detail_csv, index=False)

    summary = {
        "scenario": scenario_name,
        "sim_config": str(sim_config_path),
        "train_episodes": train_episodes,
        "train_iterations": train_iterations,
        "training_csv": str(training_csv),
        "iterations": iterations,
        "eval_interval": eval_interval,
        "n_steps": int(len(detail_df)),
        "total_violations_across_steps": int(detail_df["total_violations"].sum()),
        "mean_step_total_violations": float(detail_df["total_violations"].mean()),
        "training_mean_episode_total_violations": float(training_df["total_violations"].mean()) if not training_df.empty else 0.0,
        "agents": sorted(dispatcher.agents_by_name.keys()),
        "service_agent_mapping": {f"s{int(k)}": v for k, v in sorted(service_to_agent.items())},
    }
    for node in runtime.nodes:
        col = f"node{int(node.id)}_avg_cpu_usage_pct"
        if col in detail_df.columns:
            summary[f"{col}_mean"] = float(detail_df[col].mean())

    summary_json = exp_dir / f"evaluation_{scenario_name}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    return {
        "scenario": scenario_name,
        "training_csv": str(training_csv),
        "detail_csv": str(detail_csv),
        "summary_json": str(summary_json),
        "train_episodes": train_episodes,
        "n_steps": int(len(detail_df)),
        "total_violations": int(detail_df["total_violations"].sum()),
    }


def evaluate_allocation(args: argparse.Namespace, exp_dir: Path, sim_config_path: Path) -> None:
    print("\n=== Step 5/5: Evaluate allocation (Evaluator) ===")

    agent_yaml = exp_dir / "agent_assignments.yaml"
    selected_service_to_agent = _load_service_agent_mapping(agent_yaml)

    if str(WORKFLOW_SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKFLOW_SIM_ROOT))

    from simulator.application import Application  # pyright: ignore[reportMissingImports]
    from simulator.infrastructure import Node as SimNode  # pyright: ignore[reportMissingImports]
    from simulator.simulation import Simulation  # pyright: ignore[reportMissingImports]

    config = _strip_agent_config(load_simulator_config(sim_config_path))

    sim_cfg = config.get("simulation", {})
    reward_cfg = config.get("reward", {})
    infra_cfg = config.get("infrastructure", {})

    seed = int(args.seed if args.seed is not None else sim_cfg.get("seed", 42))
    np.random.seed(seed)

    iterations = int(args.iterations or sim_cfg.get("iterations", 500))
    timeout = int(sim_cfg.get("timeout", 600000))
    eval_interval = float(sim_cfg.get("eval_interval", 10.0))
    latency_target = reward_cfg.get("e2e_lat_target", None)
    cpu_max = int(infra_cfg.get("cpu_max", 4))
    print(f"[Step 5] Using simulator config: {sim_config_path}")
    print(f"[Step 5] Assignment file: {agent_yaml}")
    print("[Step 5] Running selected config and baseline scenarios")

    base_runtime = build_runtime(
        config=config,
        simulation_classes=(Application, SimNode),
        perturb_threads=False,
        perturb_prob=0.0,
        thread_min=1,
        thread_max=max(1, cpu_max),
    )

    baseline_mappings = _build_baseline_mappings(base_runtime)
    service_to_workflows, service_to_node = _build_service_scopes(base_runtime)

    scenarios: list[tuple[str, dict[int, str]]] = [
        ("selected", selected_service_to_agent),
        ("one_agent_all_tasks", baseline_mappings["one_agent_all_tasks"]),
        ("one_agent_per_task", baseline_mappings["one_agent_per_task"]),
        ("one_agent_per_workflow", baseline_mappings["one_agent_per_workflow"]),
        ("one_agent_per_node", baseline_mappings["one_agent_per_node"]),
    ]

    scenario_results = []
    for scenario_name, service_to_agent in scenarios:
        scenario_config_path = _write_scenario_sim_config(
            exp_dir=exp_dir,
            scenario_name=scenario_name,
            base_config=config,
            service_to_agent=service_to_agent,
            service_to_workflows=service_to_workflows,
            service_to_node=service_to_node,
        )
        print(f"[Step 5] Scenario: {scenario_name}")
        print(f"[Step 5] Scenario config: {scenario_config_path}")
        scenario_results.append(
            _run_step_5_scenario(
                args=args,
                exp_dir=exp_dir,
                scenario_name=scenario_name,
                service_to_agent=service_to_agent,
                sim_config_path=scenario_config_path,
                cpu_max=cpu_max,
                iterations=iterations,
                timeout=timeout,
                eval_interval=eval_interval,
                latency_target=latency_target,
                seed=seed,
            )
        )

    manifest = {
        "assignment_file": str(agent_yaml),
        "sim_config": str(sim_config_path),
        "scenarios": scenario_results,
    }
    manifest_json = exp_dir / "evaluation_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print(f"[OK] Saved evaluation manifest: {manifest_json}")
    print(f"[OK] Completed {len(scenario_results)} evaluation scenarios")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UML experiment sequence using exp_config.yaml.")
    parser.add_argument(
        "--exp-config",
        default="exp_config.yaml",
        help="Path to experiment YAML config (default: exp_config.yaml)",
    )
    cli_args = parser.parse_args()

    config_path = Path(cli_args.exp_config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    args = _load_exp_config(config_path)
    setattr(args, "exp_config_source", str(config_path))
    return args


def main() -> None:
    args = parse_args()

    dataset_dir = _resolve_dataset_dir(str(args.dataset_dir))
    dataset_csv, sim_config_path, dataset_config_path = _load_dataset_artifacts(dataset_dir)

    exp_dir = _create_experiment_dir(Path(args.experiments_dir), args.experiment_id)
    print(f"Experiment directory: {exp_dir}")

    setattr(args, "dataset_dir", str(dataset_dir))
    setattr(args, "dataset_csv", str(dataset_csv))
    setattr(args, "sim_config", str(sim_config_path))

    persisted_config = _persist_experiment_config(exp_dir, args)
    print(f"[OK] Saved experiment config: {persisted_config}")

    print("\n=== Dataset Input (from Stage 1 script) ===")
    print(f"[OK] Dataset directory: {dataset_dir}")
    print(f"[OK] Dataset config: {dataset_config_path}")
    print(f"[OK] Using provided dataset: {dataset_csv}")
    print(f"[OK] Using simulator config from dataset dir: {sim_config_path}")

    if args.step >= 2:
        build_causal_graph(args, exp_dir, dataset_csv)
    step3_analyzer: TaskInterferenceAnalyzer | None = None
    step3_tig: TaskInterferenceGraph | None = None
    if args.step >= 3:
        _, step3_analyzer, step3_tig = build_tig(args, exp_dir, dataset_csv)
    if args.step >= 4:
        get_partition(args, exp_dir, analyzer=step3_analyzer, tig=step3_tig)
        agent_snapshot = _persist_agent_allocation_snapshot(exp_dir)
        print(f"[OK] Saved agent allocation snapshot: {agent_snapshot}")
    if args.step >= 5:
        evaluate_allocation(args, exp_dir, sim_config_path=sim_config_path)


if __name__ == "__main__":
    main()
