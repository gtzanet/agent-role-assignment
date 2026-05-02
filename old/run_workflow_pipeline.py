import argparse
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel

from allocation import Allocator
from causality.task_interference_analyzer import compute_task_interaction_weight_matrix
from causality.causal_discovery import infer_causal_graph_notears
from environment import CausalGraph, Node, NodeType
from visualize_workflow_causal_graph import infer_and_visualize_causal_graph


def _discretize_for_inference(df: pd.DataFrame, bins: int = 5) -> tuple[pd.DataFrame, Dict[str, Dict[int, float]]]:
    discretized = pd.DataFrame(index=df.index)
    representatives: Dict[str, Dict[int, float]] = {}

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        uniq = np.sort(s.dropna().unique())

        if len(uniq) <= max(10, bins):
            vals = pd.Series(uniq)
            mapping = {int(v): float(v) for v in vals if float(v).is_integer()}
            if len(mapping) == len(uniq):
                discretized[col] = s.astype("Int64")
                representatives[col] = {int(k): float(v) for k, v in mapping.items()}
                continue

        q_bins = min(bins, max(2, len(uniq)))
        cat = pd.qcut(s, q=q_bins, labels=False, duplicates="drop")
        cat = cat.astype("Int64")
        discretized[col] = cat

        reps: Dict[int, float] = {}
        valid = pd.DataFrame({"raw": s, "bin": cat}).dropna()
        for b in sorted(valid["bin"].unique()):
            b_int = int(b)
            reps[b_int] = float(valid[valid["bin"] == b_int]["raw"].mean())
        representatives[col] = reps

    return discretized.dropna().astype(int), representatives


def _compute_interventional_delta(
    cg: CausalGraph,
    raw_df: pd.DataFrame,
    tasks: List[str],
    kpis: List[str],
    bins: int = 5,
    max_parents: int = 3,
) -> np.ndarray:
    descendants = set()
    for t in tasks:
        if t in cg.graph:
            descendants.update(nx.descendants(cg.graph, t))

    ancestors = set()
    for k in kpis:
        if k in cg.graph:
            ancestors.update(nx.ancestors(cg.graph, k))

    path_nodes = set(tasks) | set(kpis) | (descendants & ancestors)
    cols = [c for c in raw_df.columns if c in cg.nodes and c in path_nodes]
    data_df, reps = _discretize_for_inference(raw_df[cols], bins=bins)

    sm = StructureModel()
    sm.add_nodes_from([c for c in data_df.columns if c in cg.nodes])

    parents_by_child: Dict[str, List[str]] = {}
    for u, v in cg.graph.edges():
        if u in data_df.columns and v in data_df.columns:
            parents_by_child.setdefault(v, []).append(u)

    for child, parents in parents_by_child.items():
        if len(parents) <= max_parents:
            for p in parents:
                sm.add_edge(p, child)
            continue

        scored = []
        for p in parents:
            corr = data_df[p].corr(data_df[child])
            score = abs(float(corr)) if pd.notna(corr) else 0.0
            scored.append((score, p))
        scored.sort(reverse=True)
                ie.reset_do(t)

        descendants = set()
        for t in tasks:
            if t in cg.graph:
                descendants.update(nx.descendants(cg.graph, t))

        ancestors = set()
        for k in kpis:
            if k in cg.graph:
                ancestors.update(nx.ancestors(cg.graph, k))

        path_nodes = set(tasks) | set(kpis) | (descendants & ancestors)
        cols = [c for c in raw_df.columns if c in cg.nodes and c in path_nodes]
        if not cols:
            return delta

        data_df, reps = _discretize_for_inference(raw_df[cols], bins=bins)

        working_graph = cg.graph.subgraph(data_df.columns).copy()
        if working_graph.number_of_nodes() == 0:
            return delta

        components = list(nx.connected_components(working_graph.to_undirected()))

        for component in components:
            component_nodes = [c for c in data_df.columns if c in component]
            if len(component_nodes) < 2:
                continue

            component_df = data_df[component_nodes]

            sm = StructureModel()
            sm.add_nodes_from(component_nodes)

            parents_by_child: Dict[str, List[str]] = {}
            for u, v in working_graph.edges():
                if u in component_nodes and v in component_nodes:
                    parents_by_child.setdefault(v, []).append(u)

            for child, parents in parents_by_child.items():
                if len(parents) <= max_parents:
                    for p in parents:
                        sm.add_edge(p, child)
                    continue

                scored = []
                for p in parents:
                    corr = component_df[p].corr(component_df[child])
                    score = abs(float(corr)) if pd.notna(corr) else 0.0
                    scored.append((score, p))
                scored.sort(reverse=True)
                for _, p in scored[:max_parents]:
                    sm.add_edge(p, child)

            bn = BayesianNetwork(sm)
            bn = bn.fit_node_states(component_df)
            bn = bn.fit_cpds(component_df, method="BayesianEstimator", bayes_prior="K2")
            ie = InferenceEngine(bn)

            local_tasks = [t for t in tasks if t in component_nodes]
            local_kpis = [k for k in kpis if k in component_nodes]

            for i, t in enumerate(tasks):
                if t not in local_tasks:
                    continue
                if t not in component_df.columns:
                    continue

                task_vals = sorted(component_df[t].dropna().unique().tolist())
                if len(task_vals) < 2:
                    continue

                for j, k in enumerate(kpis):
                    if k not in local_kpis:
                        continue
                    if k not in component_df.columns:
                        continue

                    k_min = float(raw_df[k].min())
                    k_max = float(raw_df[k].max())
                    denom = k_max - k_min
                    if denom <= 1e-12:
                        denom = 1.0

                    effects = []
                    for x in task_vals:
                        intervention = {int(v): 0.0 for v in task_vals}
                        intervention[int(x)] = 1.0
                        ie.do_intervention(t, intervention)

                        q = ie.query({})
                        dist = q[k]

                        e_k = 0.0
                        for state, prob in dist.items():
                            state_int = int(state)
                            rep = reps.get(k, {}).get(state_int, float(state_int))
                            e_k += float(prob) * rep
                        effects.append(e_k)
                        ie.reset_do(t)

                    if effects:
                        delta[i, j] = (max(effects) - min(effects)) / denom

    return delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal discovery -> TIG generation -> graph partitioning workflow."
    )

    parser.add_argument("--c-max", type=float, default=100.0)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument(
        "--omega",
        type=str,
        default=None,
        help="Comma-separated KPI weights. If omitted, all ones are used.",
    )

    parser.add_argument(
        "--partition-algorithm",
        type=str,
        choices=["spectral", "greedy_modularity", "kernighan_lin"],
        default="spectral",
    )
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-prefix", type=str, default="workflow_run")
    parser.add_argument("--replica-action-space", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--delta-mode",
        type=str,
        choices=["interventional", "reachability"],
        default="interventional",
    )
    parser.add_argument("--delta-bins", type=int, default=5)
    parser.add_argument("--delta-max-parents", type=int, default=3)

    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    task_names = sorted([c for c in df.columns if c.endswith("_Replicas")])
    latency_kpis = sorted([c for c in df.columns if c.startswith("E2E_Latency_Workflow_")])
    cpu_kpis = sorted([c for c in df.columns if c.endswith("_Node_CPU_Usage")])
    kpi_names = sorted(set(latency_kpis + cpu_kpis))
    if not kpi_names:
        kpi_names = sorted(
            set(
                [c for c in df.columns if c.endswith("_Latency_ms")]
                + [c for c in df.columns if c.endswith("_Node_CPU_Usage")]
            )
        )

    if not task_names:
        raise RuntimeError("No task columns found. Expected columns ending with '_Replicas'.")
    if not kpi_names:
        raise RuntimeError("No KPI columns found. Expected E2E latency and/or node CPU usage columns.")

    if args.causal_algorithm == "template":
        if not args.causal_template:
            raise ValueError("--causal-template is required when --causal-algorithm template")
        template_path = Path(args.causal_template)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {args.causal_template}")

        if template_path.suffix.lower() == ".csv":
            edges_df = pd.read_csv(template_path)
            if "source" not in edges_df.columns or "target" not in edges_df.columns:
                raise ValueError("Template CSV must contain source,target[,weight] columns")
            nx_graph = nx.DiGraph()
            for _, row in edges_df.iterrows():
                w = float(row["weight"]) if "weight" in edges_df.columns else 1.0
                nx_graph.add_edge(str(row["source"]), str(row["target"]), weight=w)
        elif template_path.suffix.lower() == ".json":
            data = json.loads(template_path.read_text())
            edges = data.get("edges", data)
            nx_graph = nx.DiGraph()
            for e in edges:
                if isinstance(e, dict):
                    nx_graph.add_edge(str(e["source"]), str(e["target"]), weight=float(e.get("weight", 1.0)))
                else:
                    nx_graph.add_edge(str(e[0]), str(e[1]), weight=float(e[2]) if len(e) > 2 else 1.0)
        else:
            nx_graph = nx.read_weighted_edgelist(template_path, create_using=nx.DiGraph)

        while not nx.is_directed_acyclic_graph(nx_graph):
            cycle = nx.find_cycle(nx_graph)
            weakest = min(cycle, key=lambda e: abs(nx_graph[e[0]][e[1]].get("weight", 1.0)))
            nx_graph.remove_edge(weakest[0], weakest[1])

        cg = CausalGraph()
        for n in nx_graph.nodes():
            if n in task_names:
                cg.add_node(Node(n, NodeType.INPUT, args.replica_action_space))
            elif n in kpi_names:
                cg.add_node(Node(n, NodeType.KPI))
            else:
                cg.add_node(Node(n, NodeType.INTERMEDIARY))
        for u, v in nx_graph.edges():
            if u in cg.nodes and v in cg.nodes:
                cg.add_edge(u, v)

    elif args.causal_algorithm == "workflow_notears":
        nx_graph, _ = infer_and_visualize_causal_graph(args.dataset, threshold=args.causal_threshold)
        cg = CausalGraph()
        for n in nx_graph.nodes():
            if n in task_names:
                cg.add_node(Node(n, NodeType.INPUT, args.replica_action_space))
            elif n in kpi_names:
                cg.add_node(Node(n, NodeType.KPI))
            else:
                cg.add_node(Node(n, NodeType.INTERMEDIARY))
        for u, v in nx_graph.edges():
            if u in cg.nodes and v in cg.nodes:
                cg.add_edge(u, v)

    else:
        node_types = {
            name: (
                NodeType.INPUT if name in task_names else
                NodeType.KPI if name in kpi_names else
                NodeType.INTERMEDIARY
            )
            for name in df.columns
        }
        decision_space_sizes = {name: args.replica_action_space for name in task_names}
        cg = infer_causal_graph_notears(
            args.dataset,
            node_types=node_types,
            threshold=args.causal_threshold,
            decision_space_sizes=decision_space_sizes,
        )

        # Keep only replica tasks for this workflow pipeline
        for input_node in cg.get_inputs():
            if input_node.name not in task_names:
                cg.nodes[input_node.name].node_type = NodeType.INTERMEDIARY
            else:
                cg.nodes[input_node.name].decision_space_size = args.replica_action_space
        for kpi in kpi_names:
            if kpi in cg.nodes:
                cg.nodes[kpi].node_type = NodeType.KPI

    if args.causal_template and args.causal_algorithm != "template":
        print("[WARN] --causal-template provided but ignored because causal algorithm is not 'template'.")

    tasks = [n.name for n in cg.get_inputs() if n.name in task_names]
    kpis = [n.name for n in cg.get_kpis() if n.name in kpi_names]
    if not tasks:
        tasks = task_names
    if not kpis:
        kpis = kpi_names

    action_space = np.array([args.replica_action_space for _ in tasks], dtype=float)
    if args.omega:
        omega = np.array([float(x.strip()) for x in args.omega.split(",") if x.strip()], dtype=float)
        if len(omega) != len(kpis):
            raise ValueError(f"--omega must contain exactly {len(kpis)} values")
    else:
        omega = np.ones(len(kpis), dtype=float)

    if args.delta_mode == "interventional":
        delta = _compute_interventional_delta(
            cg,
            df,
            tasks,
            kpis,
            bins=args.delta_bins,
            max_parents=args.delta_max_parents,
        )
    else:
        delta = np.array(
            [[1.0 if k in cg.get_reachable_kpis(t) else 0.0 for k in kpis] for t in tasks],
            dtype=float,
        )

    # Default TIG equations (requested):
    # causal vectors -> cosine pull -> logistic push -> W = F_pull * (1 - F_push)
    if args.tig_pull_metric == "cosine" and args.tig_push_metric == "logistic":
        w = compute_task_interaction_weight_matrix(delta, omega, action_space, args.c_max, args.rho)
    # Keep existing project path as an alternative (legacy Allocator metrics)
    elif args.tig_pull_metric == "jaccard" and args.tig_push_metric == "linear":
        allocator = Allocator(cg)
        allocator.build_interaction_graph(
            alpha=args.alpha,
            beta=args.beta,
            complexity_limit=int(args.c_max),
        )
        w = nx.to_numpy_array(allocator.tig, nodelist=tasks, weight="weight")
    # Additional alternatives while preserving the same matrix style
    else:
        causal_vectors = delta * omega

        if args.tig_pull_metric == "cosine":
            norms = np.linalg.norm(causal_vectors, axis=1)
            denom = np.outer(norms, norms)
            dot = causal_vectors @ causal_vectors.T
            with np.errstate(divide="ignore", invalid="ignore"):
                f_pull = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        elif args.tig_pull_metric == "jaccard":
            b = (causal_vectors > 0).astype(float)
            inter = b @ b.T
            counts = b.sum(axis=1)
            union = np.add.outer(counts, counts) - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                f_pull = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        else:
            raise ValueError(f"Unsupported pull metric: {args.tig_pull_metric}")

        c = np.outer(action_space, action_space)
        if args.tig_push_metric == "logistic":
            x = np.clip(args.rho * (c - args.c_max), -700, 700)
            f_push = 1.0 / (1.0 + np.exp(-x))
        elif args.tig_push_metric == "linear":
            f_push = np.clip(c / max(args.c_max, 1e-12), 0.0, 1.0)
        elif args.tig_push_metric == "none":
            f_push = np.zeros_like(c)
        else:
            raise ValueError(f"Unsupported push metric: {args.tig_push_metric}")

        w = f_pull * (1.0 - f_push)
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)

    if len(tasks) <= args.n_agents:
        partitions: Dict[int, List[str]] = {i: [task] for i, task in enumerate(tasks)}
    elif args.partition_algorithm == "spectral":
        labels = SpectralClustering(
            n_clusters=args.n_agents,
            affinity="precomputed",
            random_state=args.seed,
            assign_labels="discretize",
        ).fit_predict(w)
        partitions = {}
        for idx, label in enumerate(labels):
            partitions.setdefault(int(label), []).append(tasks[idx])
    else:
        ug = nx.Graph()
        ug.add_nodes_from(tasks)
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if w[i, j] > 0:
                    ug.add_edge(tasks[i], tasks[j], weight=float(w[i, j]))

        if args.partition_algorithm == "greedy_modularity":
            communities = list(nx.algorithms.community.greedy_modularity_communities(ug, weight="weight"))
            partitions = {i: sorted(list(c)) for i, c in enumerate(communities)}
        elif args.partition_algorithm == "kernighan_lin":
            if args.n_agents != 2:
                raise ValueError("kernighan_lin requires --n-agents 2")
            left, right = nx.algorithms.community.kernighan_lin_bisection(ug, weight="weight", seed=args.seed)
            partitions = {0: sorted(list(left)), 1: sorted(list(right))}
        else:
            raise ValueError(f"Unsupported partition algorithm: {args.partition_algorithm}")

    prefix = args.output_prefix
    np.savetxt(f"{prefix}_delta.csv", delta, delimiter=",", fmt="%.6f")
    np.savetxt(f"{prefix}_W.csv", w, delimiter=",", fmt="%.6f")

    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks": tasks,
                "kpis": kpis,
                "causal_algorithm": args.causal_algorithm,
                "tig_pull_metric": args.tig_pull_metric,
                "tig_push_metric": args.tig_push_metric,
                "partition_algorithm": args.partition_algorithm,
                "partitions": partitions,
            },
            f,
            indent=2,
        )

    print("=== Workflow completed ===")
    print(f"Tasks ({len(tasks)}): {tasks}")
    print(f"KPIs ({len(kpis)}): {kpis}")
    print("Partitions:")
    for pid, members in partitions.items():
        print(f"  Agent {pid}: {members}")
    print(f"Saved: {prefix}_delta.csv, {prefix}_W.csv, {prefix}_meta.json")


if __name__ == "__main__":
    main()
