"""Flask server hosting CausalAnalyzer for remote causal inference.

Start with:
    python -m causality.causal_inference_server \\
        --dataset-dir data/my_dataset \\
        --workflow-config workflow_causal_graph.yaml \\
        [--causal-algorithm correlation] \\
        [--causal-threshold 0.1] \\
        [--bins 5] \\
        [--max-parents 3] \\
        [--host 0.0.0.0] \\
        [--port 5050]

The causal graph and Bayesian Network are built from the dataset at startup.
Inference endpoints are available once the server prints "Starting server...".
The server is stateful and not thread-safe — run with a single worker (default).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from flask import Flask, jsonify, request

from causality.causal_discovery import (
    infer_causal_graph_correlation,
    infer_causal_graph_lingam,
    infer_causal_graph_notears,
)
from causality.causal_inference import CausalAnalyzer
from environment import NodeType

app = Flask(__name__)

# Module-level state — populated by _build() before Flask starts
_analyzer: CausalAnalyzer | None = None


def _serialize_df(df: pd.DataFrame) -> str:
    return df.to_json(orient="split")


def _serialize_representatives(reps: dict) -> dict:
    return {col: {str(k): v for k, v in bins.items()} for col, bins in reps.items()}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/prepare_from_observations", methods=["POST"])
def prepare_from_observations():
    """Return pre-built BN state filtered to the requested columns.

    The dataset, causal graph, and BN are fixed at startup.
    This endpoint does not re-fit the network — it only filters the cached
    state so the client knows which components and nodes are relevant.
    """
    data = request.get_json()
    requested: list[str] = data.get("columns", [])

    avail = (
        [c for c in requested if c in _analyzer.data_df.columns]
        if requested
        else list(_analyzer.data_df.columns)
    )
    avail_set = set(avail)

    data_df = _analyzer.data_df[avail] if avail else _analyzer.data_df
    reps = {k: v for k, v in _analyzer.representatives.items() if k in avail_set}
    n2c = {k: int(v) for k, v in _analyzer.node_to_component.items() if k in avail_set}
    comp = {
        str(idx): [n for n in nodes if n in avail_set]
        for idx, (nodes, _, _) in _analyzer.component_models.items()
        if any(n in avail_set for n in nodes)
    }

    return jsonify({
        "data_df": _serialize_df(data_df),
        "representatives": _serialize_representatives(reps),
        "node_to_component": n2c,
        "component_nodes_by_idx": comp,
    })


@app.route("/interventional_effects_for_task", methods=["POST"])
def interventional_effects_for_task():
    data = request.get_json()
    effects = _analyzer.interventional_effects_for_task(
        task_node=data["task_node"],
        kpi_nodes=data["kpi_nodes"],
        raw_ranges={k: tuple(v) for k, v in data["raw_ranges"].items()},
        component_idx=data.get("component_idx"),
    )
    return jsonify({"effects": effects})


@app.route("/interventional_effect", methods=["POST"])
def interventional_effect():
    data = request.get_json()
    effect = _analyzer.interventional_effect(
        task_node=data["task_node"],
        kpi_node=data["kpi_node"],
        raw_min=float(data["raw_min"]),
        raw_max=float(data["raw_max"]),
        component_idx=data.get("component_idx"),
        task_vals=data.get("task_vals"),
    )
    return jsonify({"effect": effect})


@app.route("/observational_query", methods=["POST"])
def observational_query():
    evidence = request.get_json().get("evidence")
    results = _analyzer.observational_query(evidence=evidence)
    return jsonify({
        "distributions": {
            var: {str(k): float(v) for k, v in dist.items()}
            for var, dist in results.items()
        }
    })


@app.route("/counterfactual_query", methods=["POST"])
def counterfactual_query():
    data = request.get_json()
    results = _analyzer.counterfactual_query(
        evidence={k: int(v) for k, v in data["evidence"].items()},
        intervention={k: int(v) for k, v in data["intervention"].items()},
    )
    return jsonify({
        "distributions": {
            var: {str(k): float(v) for k, v in dist.items()}
            for var, dist in results.items()
        }
    })


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _load_dataset(dataset_dir: Path) -> tuple[Path, pd.DataFrame]:
    dataset_config_path = dataset_dir / "dataset_config.yaml"
    filename = "dataset.csv"
    if dataset_config_path.exists():
        cfg = yaml.safe_load(dataset_config_path.read_text()) or {}
        filename = str(cfg.get("dataset_filename", filename))
    csv_path = dataset_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    print(f"[INFO] Loading dataset: {csv_path}")
    return csv_path, pd.read_csv(csv_path)


def _load_node_metadata(
    workflow_config_path: Path, columns: list[str]
) -> tuple[dict[str, NodeType], dict[str, int]]:
    config = yaml.safe_load(workflow_config_path.read_text()) or {}
    cg_cfg = config.get("causal_graph", {})
    node_types_raw = cg_cfg.get("node_types", {})
    decision_space_raw = cg_cfg.get("decision_space_sizes", {})
    node_types: dict[str, NodeType] = {}
    decision_spaces: dict[str, int] = {}
    for col in columns:
        if col not in node_types_raw:
            raise ValueError(
                f"Missing node type for column '{col}' in {workflow_config_path}"
            )
        node_types[col] = NodeType(str(node_types_raw[col]).strip().lower())
        decision_spaces[col] = int(decision_space_raw.get(col, 1))
    return node_types, decision_spaces


def _build(args: argparse.Namespace) -> None:
    """Run causal discovery and fit the BN. Called once before Flask starts."""
    global _analyzer

    dataset_dir = Path(args.dataset_dir).resolve()
    workflow_config_path = Path(args.workflow_config).resolve()

    dataset_csv, raw_df = _load_dataset(dataset_dir)
    columns = list(raw_df.columns)
    node_types, decision_spaces = _load_node_metadata(workflow_config_path, columns)

    print(
        f"[INFO] Running causal discovery "
        f"(algorithm={args.causal_algorithm}, threshold={args.causal_threshold})..."
    )
    if args.causal_algorithm == "notears":
        cg = infer_causal_graph_notears(
            str(dataset_csv), node_types, args.causal_threshold, decision_spaces
        )
    elif args.causal_algorithm == "lingam":
        cg = infer_causal_graph_lingam(
            str(dataset_csv), node_types, args.causal_threshold, decision_spaces
        )
    else:
        cg = infer_causal_graph_correlation(
            str(dataset_csv), node_types, args.causal_threshold, decision_spaces
        )

    print(f"[INFO] Building Bayesian Network (bins={args.bins}, max_parents={args.max_parents})...")
    _analyzer = CausalAnalyzer(cg)
    _analyzer.prepare_from_observations(raw_df, columns, bins=args.bins, max_parents=args.max_parents)

    n_components = len(_analyzer.component_models)
    n_nodes = len(_analyzer.node_to_component)
    print(f"[INFO] BN ready: {n_components} component(s), {n_nodes} fitted nodes")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CausalAnalyzer inference server")
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Directory containing the dataset CSV and dataset_config.yaml",
    )
    parser.add_argument(
        "--workflow-config", required=True,
        help="Path to workflow YAML defining node types and decision space sizes",
    )
    parser.add_argument(
        "--causal-algorithm", default="correlation",
        choices=["notears", "correlation", "lingam"],
        help="Causal discovery algorithm (default: correlation)",
    )
    parser.add_argument(
        "--causal-threshold", type=float, default=0.1,
        help="Edge-weight threshold for causal discovery (default: 0.1)",
    )
    parser.add_argument(
        "--bins", type=int, default=5,
        help="Number of quantile bins for BN discretization (default: 5)",
    )
    parser.add_argument(
        "--max-parents", type=int, default=3,
        help="Maximum parents per node in the BN (default: 3)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _build(args)
    print(f"[INFO] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
