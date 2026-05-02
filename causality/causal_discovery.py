
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from typing import Mapping
from causalnex.structure.notears import from_pandas

from environment import CausalGraph, Node, NodeType


def _node_type_to_tier(node_type: NodeType) -> int:
    if node_type == NodeType.INPUT:
        return 0
    if node_type == NodeType.INTERMEDIARY:
        return 1
    if node_type == NodeType.KPI:
        return 2
    raise ValueError(f"Unsupported node type: {node_type}")


def _normalize_node_metadata(
    columns: list[str],
    node_types: Mapping[str, NodeType],
    decision_space_sizes: Mapping[str, int] | None = None,
) -> dict[str, tuple[NodeType, int]]:
    missing = [name for name in columns if name not in node_types]
    if missing:
        raise ValueError(
            "Missing node types for dataset columns: " + ", ".join(sorted(missing))
        )

    metadata: dict[str, tuple[NodeType, int]] = {}
    for name in columns:
        node_type = node_types[name]
        decision_space_size = 1
        if decision_space_sizes is not None:
            decision_space_size = int(decision_space_sizes.get(name, 1))
        metadata[name] = (node_type, decision_space_size)

    return metadata


def infer_causal_graph_notears(
    dataset_path: str,
    node_types: Mapping[str, NodeType],
    threshold: float = 0.1,
    decision_space_sizes: Mapping[str, int] | None = None,
) -> CausalGraph:
    """Infer a causal graph using CausalNex NOTEARS with hard node-type constraints.

    Constraints enforced via tabu edges:
    - Inputs: only outgoing edges; no incoming edges.
    - KPIs: only incoming edges; no outgoing edges.
    - Intermediaries: can connect according to tier ordering with inputs/KPIs.
    """
    print(f"[INFO] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    labels = df.columns.tolist()

    metadata = _normalize_node_metadata(labels, node_types, decision_space_sizes)

    tiers = {}
    for label in labels:
        tiers[label] = _node_type_to_tier(metadata[label][0])

    tabu_edges = []
    for source in labels:
        for target in labels:
            if source == target:
                continue
            source_type = metadata[source][0]
            target_type = metadata[target][0]

            # Keep directional tier constraint (no edges from higher tier to lower tier).
            if tiers[source] > tiers[target]:
                tabu_edges.append((source, target))
                continue

            # Inputs can only have outgoing edges.
            if target_type == NodeType.INPUT:
                tabu_edges.append((source, target))
                continue

            # KPIs can only have incoming edges.
            if source_type == NodeType.KPI:
                tabu_edges.append((source, target))
                continue

    print(f"[INFO] Running CausalNex (from_pandas) with {len(tabu_edges)} tabu edges...")
    sm = from_pandas(df, tabu_edges=tabu_edges)

    print(f"[INFO] Filtering edges with weight < {threshold}...")
    sm.remove_edges_below_threshold(threshold)

    graph_nx = nx.DiGraph(sm)

    while not nx.is_directed_acyclic_graph(graph_nx):
        try:
            cycle = nx.find_cycle(graph_nx)
            weakest_edge = min(cycle, key=lambda edge: graph_nx[edge[0]][edge[1]]["weight"])
            print(
                f"[WARN] Cycle detected. Removing weakest edge: {weakest_edge} "
                f"(w={graph_nx[weakest_edge[0]][weakest_edge[1]]['weight']:.4f})"
            )
            graph_nx.remove_edge(*weakest_edge)
        except nx.NetworkXNoCycle:
            break

    graph = CausalGraph()
    for label in labels:
        node_type, decision_space_size = metadata[label]
        graph.add_node(Node(label, node_type, decision_space_size))

    print("[INFO] Inferred Edges (NOTEARS + Granular Tabu):")
    for u, v, data in graph_nx.edges(data=True):
        weight = data.get("weight", 0)
        print(f"  [ADD] {u} -> {v} (w={weight:.4f})")
        graph.add_edge(u, v)

    return graph


def infer_causal_graph_correlation(
    dataset_path: str,
    node_types: Mapping[str, NodeType],
    threshold: float = 0.2,
    decision_space_sizes: Mapping[str, int] | None = None,
) -> CausalGraph:
    """Fast causal proxy using correlation strength plus tier-based direction."""
    df = pd.read_csv(dataset_path)
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all").dropna()
    cols = list(df.columns)

    metadata = _normalize_node_metadata(cols, node_types, decision_space_sizes)

    graph = CausalGraph()
    tier = {}
    for col in cols:
        node_type, decision_space = metadata[col]
        tier[col] = _node_type_to_tier(node_type)
        graph.add_node(Node(col, node_type, decision_space))

    corr = df.corr(numeric_only=True)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            c = float(corr.loc[a, b]) if pd.notna(corr.loc[a, b]) else 0.0
            if abs(c) < threshold:
                continue

            ta = tier[a]
            tb = tier[b]
            if ta < tb:
                src, dst = a, b
            elif tb < ta:
                src, dst = b, a
            else:
                src, dst = (a, b) if a < b else (b, a)

            if src in graph.nodes and dst in graph.nodes and src != dst:
                graph.add_edge(src, dst)

    return graph


def infer_causal_graph_lingam(
    dataset_path: str,
    node_types: Mapping[str, NodeType],
    threshold: float = 0.1,
    decision_space_sizes: Mapping[str, int] | None = None,
) -> CausalGraph:
    """Causal discovery using DirectLiNGAM with edge-weight thresholding."""
    try:
        from lingam import DirectLiNGAM
    except ImportError as exc:
        raise ImportError(
            "LiNGAM is not installed in the active environment. "
            "Install it with: pip install lingam"
        ) from exc

    df = pd.read_csv(dataset_path)
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all").dropna()
    all_cols = list(df.columns)
    if not all_cols:
        raise RuntimeError("No numeric columns available for LiNGAM causal discovery.")

    metadata = _normalize_node_metadata(all_cols, node_types, decision_space_sizes)

    non_constant_cols = [c for c in all_cols if float(df[c].var()) > 1e-12]
    if len(non_constant_cols) < 2:
        raise RuntimeError(
            "LiNGAM requires at least two non-constant numeric columns. "
            "Collect more variable data before running --causal-algorithm lingam."
        )

    n_samples = len(df)
    n_features = len(non_constant_cols)
    if n_samples <= n_features:
        raise RuntimeError(
            "LiNGAM requires sufficient data: number of samples must be greater than "
            f"number of non-constant features (samples={n_samples}, features={n_features}). "
            "Rerun Step 1 with a longer early-stop window and/or higher --max-samples."
        )

    model = DirectLiNGAM(random_state=42)
    model.fit(df[non_constant_cols].values)
    adj = np.asarray(model.adjacency_matrix_, dtype=float)

    graph = CausalGraph()
    for col in all_cols:
        node_type, decision_space = metadata[col]
        graph.add_node(Node(col, node_type, decision_space))

    for i, dst in enumerate(non_constant_cols):
        for j, src in enumerate(non_constant_cols):
            if i == j:
                continue
            w = float(adj[i, j])
            if abs(w) >= threshold and src in graph.nodes and dst in graph.nodes:
                graph.add_edge(src, dst)

    return graph