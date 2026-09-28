"""Reusable partitioning helpers for analytical GPS configs.

This module is intentionally importable from scripts and evaluators so the
partitioning logic can be accessed independently of the CLI entry point in
partition_tig.py.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities, kernighan_lin_bisection
from sklearn.cluster import SpectralClustering

from gps_model import GPSModel

TIG_ALGOS = ["greedy_modularity", "spectral", "kernighan_lin"]
TOPO_ALGOS = ["all_in_one", "all_separate", "per_node", "per_workflow"]
ALL_ALGOS = TIG_ALGOS + TOPO_ALGOS


def _sort_communities(communities: List[List[int]]) -> List[List[int]]:
    return [sorted(community) for community in sorted(communities, key=lambda community: min(community))]


def partition_task_ids(
    model: GPSModel,
    algorithm: str,
    n_agents: int,
    w: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Partition model task IDs into agent groups.

    For TIG algorithms (greedy_modularity, spectral, kernighan_lin), a weight
    matrix ``w`` is required. Topology baselines use the model structure alone.
    """

    task_ids = list(model.task_ids)
    n_tasks = len(task_ids)
    if n_tasks == 0:
        raise ValueError("Model has no tasks to partition.")

    if algorithm == "all_in_one":
        return {0: task_ids}

    if algorithm == "all_separate":
        return {index: [task_id] for index, task_id in enumerate(task_ids)}

    if algorithm == "per_node":
        groups: Dict[int, List[int]] = defaultdict(list)
        for task_id in task_ids:
            groups[int(model.services[task_id]["node"])].append(task_id)
        return {index: sorted(group) for index, group in enumerate(sorted(groups.values(), key=min))}

    if algorithm == "per_workflow":
        groups = defaultdict(list)
        for task_id in task_ids:
            groups[int(model.services[task_id]["workflow"])].append(task_id)
        return {index: sorted(group) for index, group in enumerate(sorted(groups.values(), key=min))}

    if n_tasks <= n_agents:
        return {index: [task_id] for index, task_id in enumerate(task_ids)}

    if w is None:
        raise ValueError(f"Algorithm '{algorithm}' requires a weight matrix.")

    w = np.asarray(w, dtype=float)
    if w.shape != (n_tasks, n_tasks):
        raise ValueError(f"Weight matrix shape {w.shape} does not match task count {n_tasks}.")

    if algorithm == "greedy_modularity":
        if w.sum() == 0:
            return {index: [task_id] for index, task_id in enumerate(task_ids)}
        graph = nx.from_numpy_array(w)
        communities = list(greedy_modularity_communities(graph, weight="weight"))
        return {index: [task_ids[i] for i in community] for index, community in enumerate(_sort_communities([list(c) for c in communities]))}

    if algorithm == "spectral":
        clustering = SpectralClustering(
            n_clusters=n_agents,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed if seed is not None else 0,
        )
        labels = clustering.fit_predict(w)
        communities = [[index for index, label in enumerate(labels) if label == cluster] for cluster in range(n_agents)]
        communities = [community for community in communities if community]
        communities = _sort_communities(communities)
        return {index: [task_ids[i] for i in community] for index, community in enumerate(communities)}

    if algorithm == "kernighan_lin":
        if n_agents != 2:
            raise ValueError("kernighan_lin requires n_agents = 2")
        graph = nx.from_numpy_array(w)
        left, right = kernighan_lin_bisection(graph, weight="weight", seed=seed)
        return {0: [task_ids[i] for i in sorted(left)], 1: [task_ids[i] for i in sorted(right)]}

    raise ValueError(f"Unknown algorithm: {algorithm}")