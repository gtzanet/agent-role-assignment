#!/usr/bin/env python3
"""Benchmark TIG interventional delta computation: 1 worker vs N workers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from causality.task_interference_analyzer import TaskInterferenceAnalyzer
from environment import CausalGraph, Node, NodeType
import json


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
        cg.add_node(Node(str(nd["name"]), nt, int(nd.get("decision_space_size", 1))))
    for edge in payload.get("edges", []):
        if len(edge) == 2:
            src, dst = str(edge[0]), str(edge[1])
            if src in cg.nodes and dst in cg.nodes:
                cg.add_edge(src, dst)
    return cg


def _load_variable_groups(workflow_config_path: Path, columns: list[str]) -> dict[str, list[str]]:
    config = yaml.safe_load(workflow_config_path.read_text()) or {}
    node_types_config = config.get("causal_graph", {}).get("node_types", {})
    type_map = {
        "input": NodeType.INPUT,
        "intermediary": NodeType.INTERMEDIARY,
        "kpi": NodeType.KPI,
    }
    groups: dict[str, list[str]] = {"inputs": [], "intermediates": [], "outputs": []}
    for name in columns:
        raw = str(node_types_config.get(name, "intermediary")).lower()
        nt = type_map.get(raw, NodeType.INTERMEDIARY)
        if nt == NodeType.INPUT:
            groups["inputs"].append(name)
        elif nt == NodeType.INTERMEDIARY:
            groups["intermediates"].append(name)
        elif nt == NodeType.KPI:
            groups["outputs"].append(name)
    return groups


def run_tig(df: pd.DataFrame, cg: CausalGraph, variable_groups: dict, max_workers: int) -> dict:
    analyzer = TaskInterferenceAnalyzer(cg, variable_groups)
    t0 = time.perf_counter()
    tig = analyzer.generate_tig(
        df,
        output_dir=None,
        dataset_path="data/wf02/dataset.csv",
        causal_threshold=0.1,
        delta_mode="interventional",
        bins=5,
        max_parents=3,
        c_max=100.0,
        rho=0.1,
        save_plot=False,
        max_workers=max_workers,
    )
    total = time.perf_counter() - t0
    timing = tig.timing or {}
    return {
        "max_workers": max_workers,
        "total_wall_s": round(total, 2),
        "prepare_s": round(float(timing.get("prepare_s", 0)), 2),
        "query_s": round(float(timing.get("query_s", 0)), 2),
        "reachable_pairs": timing.get("reachable_pairs", "?"),
    }


def main() -> None:
    dataset_csv = REPO_ROOT / "data" / "wf02" / "dataset.csv"
    graph_json = REPO_ROOT / "experiments" / "exp_wf02" / "causal_graph.json"
    workflow_config = REPO_ROOT / "workflow_causal_graph.yaml"

    print(f"Dataset : {dataset_csv}")
    print(f"Graph   : {graph_json}")
    print(f"Config  : {workflow_config}\n")

    df = pd.read_csv(dataset_csv)
    print(f"Loaded dataset: {len(df)} rows × {len(df.columns)} cols\n")

    cg = _deserialize_causal_graph(graph_json)
    variable_groups = _load_variable_groups(workflow_config, list(df.columns))

    print(f"Tasks (inputs) : {variable_groups['inputs']}")
    print(f"KPIs (outputs) : {variable_groups['outputs']}\n")

    results = []
    for n in [1, 6]:
        print(f"{'='*50}")
        print(f"Running TIG with max_workers={n} ...")
        print(f"{'='*50}")
        r = run_tig(df, cg, variable_groups, max_workers=n)
        results.append(r)
        print(f"\n  prepare={r['prepare_s']}s  query={r['query_s']}s  total={r['total_wall_s']}s\n")

    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    r1, r6 = results[0], results[1]
    speedup = r1["query_s"] / r6["query_s"] if r6["query_s"] > 0 else float("inf")
    wall_speedup = r1["total_wall_s"] / r6["total_wall_s"] if r6["total_wall_s"] > 0 else float("inf")
    print(f"{'':20s} {'1 worker':>12s} {'6 workers':>12s} {'speedup':>10s}")
    print(f"{'prepare (BN fit)':20s} {r1['prepare_s']:>11.2f}s {r6['prepare_s']:>11.2f}s {'N/A':>10s}")
    print(f"{'query (inference)':20s} {r1['query_s']:>11.2f}s {r6['query_s']:>11.2f}s {speedup:>9.2f}x")
    print(f"{'total wall time':20s} {r1['total_wall_s']:>11.2f}s {r6['total_wall_s']:>11.2f}s {wall_speedup:>9.2f}x")
    print(f"\nReachable pairs: {r1['reachable_pairs']}")


if __name__ == "__main__":
    main()
