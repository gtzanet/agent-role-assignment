from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import SpectralClustering

from causality.causal_inference import CausalAnalyzer, _compute_effects_worker
from environment import CausalGraph, NodeType


class _TerminalProgressBar:
    """Minimal terminal progress bar for long-running TIG calculations."""

    def __init__(self, total: int, label: str = "Progress"):
        self.total = max(int(total), 1)
        self.label = label
        self.current = 0
        self.start_time = time.time()
        self._last_render = 0.0
        self._render(force=True)

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + max(int(step), 0))
        now = time.time()
        if now - self._last_render >= 0.2 or self.current >= self.total:
            self._render(force=True)

    def close(self) -> None:
        if self.current < self.total:
            self.current = self.total
            self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, force: bool = False) -> None:
        if not force:
            return

        elapsed = max(time.time() - self.start_time, 1e-9)
        ratio = float(self.current) / float(self.total)
        percent = int(ratio * 100)
        bar_width = 28
        filled = int(ratio * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)

        rate = self.current / elapsed
        remaining = max(self.total - self.current, 0)
        eta = remaining / rate if rate > 1e-9 else float("inf")

        eta_text = "--:--"
        if eta != float("inf"):
            eta_m = int(eta // 60)
            eta_s = int(eta % 60)
            eta_text = f"{eta_m:02d}:{eta_s:02d}"

        msg = (
            f"\r{self.label}: [{bar}] {percent:3d}% "
            f"({self.current}/{self.total}) elapsed={int(elapsed)}s eta={eta_text}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
        self._last_render = time.time()


def compute_task_interaction_weight_matrix(
    delta: np.ndarray,
    omega: np.ndarray,
    A: np.ndarray,
    C_max: float,
    rho: float,
) -> np.ndarray:
    """Compute the N x N Task Interaction Weight Matrix (W)."""
    delta = np.asarray(delta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    A = np.asarray(A, dtype=float)

    if delta.ndim != 2:
        raise ValueError("delta must be a 2D array with shape (N, M).")

    N, M = delta.shape

    if omega.ndim != 1 or omega.shape[0] != M:
        raise ValueError("omega must be a 1D array with length M (delta.shape[1]).")

    if A.ndim != 1 or A.shape[0] != N:
        raise ValueError("A must be a 1D array with length N (delta.shape[0]).")

    # Causal vectors: c_i = omega ⊙ delta[i]
    causal_vectors = delta * omega

    # Pull matrix: cosine similarity of all causal vectors
    norms = np.linalg.norm(causal_vectors, axis=1)
    denom = np.outer(norms, norms)
    dot = causal_vectors @ causal_vectors.T

    with np.errstate(divide="ignore", invalid="ignore"):
        F_pull = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)

    # Joint complexity matrix: C[i, j] = A[i] * A[j]
    C = np.outer(A, A)

    # Push matrix: logistic penalty
    x = np.clip(rho * (C - C_max), -700, 700)
    F_push = 1.0 / (1.0 + np.exp(-x))

    # Final matrix: W = F_pull * (1 - F_push)
    W = F_pull * (1.0 - F_push)

    # Enforce symmetry and remove self-loops
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    return W


@dataclass
class TaskInterferenceGraph:
    tasks: list[str]
    kpis: list[str]
    delta: np.ndarray
    omega: np.ndarray
    action_space: np.ndarray
    w: np.ndarray
    dataset_path: str | None = None
    causal_threshold: float | None = None
    delta_mode: str = "interventional"
    delta_bins: int = 5
    delta_max_parents: int = 3
    c_max: float = 100.0
    rho: float = 0.1
    timing: dict[str, float | int] | None = None


class TaskInterferenceAnalyzer:
    def __init__(self, causal_graph: CausalGraph, variable_groups: dict[str, Iterable[str]]):
        self.cg = causal_graph
        self.variable_groups = self._normalize_variable_groups(variable_groups)
        self.causal_analyzer = CausalAnalyzer(causal_graph)
        self.last_tig: TaskInterferenceGraph | None = None
        self._last_delta_timing: dict[str, float | int] = {}

    @staticmethod
    def _normalize_variable_groups(variable_groups: dict[str, Iterable[str]]) -> dict[str, list[str]]:
        aliases = {
            "inputs": "inputs",
            "input": "inputs",
            "intermediates": "intermediates",
            "intermediate": "intermediates",
            "outputs": "outputs",
            "output": "outputs",
            "kpis": "outputs",
            "kpi": "outputs",
        }

        normalized: dict[str, list[str]] = {"inputs": [], "intermediates": [], "outputs": []}
        for key, values in variable_groups.items():
            canonical = aliases.get(str(key).strip().lower())
            if canonical is None:
                continue
            normalized[canonical].extend(str(value) for value in values)

        for key in normalized:
            seen: set[str] = set()
            ordered: list[str] = []
            for name in normalized[key]:
                if name in seen:
                    continue
                seen.add(name)
                ordered.append(name)
            normalized[key] = ordered

        return normalized

    def _select_relevant_columns(self, raw_df: pd.DataFrame, tasks: list[str], kpis: list[str]) -> list[str]:
        descendants: set[str] = set()
        for task in tasks:
            if task in self.cg.graph:
                descendants.update(nx.descendants(self.cg.graph, task))

        ancestors: set[str] = set()
        for kpi in kpis:
            if kpi in self.cg.graph:
                ancestors.update(nx.ancestors(self.cg.graph, kpi))

        path_nodes = set(tasks) | set(kpis) | (descendants & ancestors)
        return [column for column in raw_df.columns if column in self.cg.nodes and column in path_nodes]

    def _compute_interventional_delta(
        self,
        raw_df: pd.DataFrame,
        tasks: list[str],
        kpis: list[str],
        bins: int = 5,
        max_parents: int = 3,
        max_workers: int | None = None,
        output_dir: Path | None = None,
    ) -> np.ndarray:
        t_start = time.perf_counter()
        def _mem_snapshot(stage: str) -> None:
            try:
                p = psutil.Process()
                rss = getattr(p.memory_info(), "rss", None)
                vms = getattr(p.memory_info(), "vms", None)
                vm = psutil.virtual_memory()
                line = f"{datetime.utcnow().isoformat()}Z,{stage},{rss},{vms},{vm.total},{vm.available}\n"
                if output_dir is not None:
                    out = Path(output_dir) / "memory_profile.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with out.open("a", encoding="utf-8") as fh:
                        fh.write(line)
            except Exception:
                pass

        _mem_snapshot("delta_start")
        delta = np.zeros((len(tasks), len(kpis)), dtype=float)

        cols = self._select_relevant_columns(raw_df, tasks, kpis)
        if not cols:
            return delta

        data_df, _, component_models = self.causal_analyzer.prepare_from_observations(
            raw_df,
            cols,
            bins=bins,
            max_parents=max_parents,
        )
        _mem_snapshot("after_prepare_from_observations")
        t_after_prepare = time.perf_counter()

        if data_df.empty or not component_models:
            self._last_delta_timing = {
                "prepare_s": t_after_prepare - t_start,
                "query_s": 0.0,
                "total_s": time.perf_counter() - t_start,
                "candidate_pairs": 0,
                "reachable_pairs": 0,
                "skipped_unreachable": 0,
            }
            return delta

        reachable_kpis_by_task: dict[str, set[str]] = {}
        for task in tasks:
            if task in self.cg.nodes:
                reachable_kpis_by_task[task] = self.cg.get_reachable_kpis(task)
            else:
                reachable_kpis_by_task[task] = set()

        candidate_pairs = 0
        valid_pairs: list[tuple[int, int, str, str]] = []
        for i, task in enumerate(tasks):
            if task not in data_df.columns:
                continue
            for j, kpi in enumerate(kpis):
                candidate_pairs += 1
                if kpi not in data_df.columns:
                    continue
                if kpi not in reachable_kpis_by_task.get(task, set()):
                    continue
                valid_pairs.append((i, j, task, kpi))

        if not valid_pairs:
            self._last_delta_timing = {
                "prepare_s": t_after_prepare - t_start,
                "query_s": 0.0,
                "total_s": time.perf_counter() - t_start,
                "candidate_pairs": int(candidate_pairs),
                "reachable_pairs": 0,
                "skipped_unreachable": int(candidate_pairs),
            }
            return delta

        progress = _TerminalProgressBar(
            total=len(valid_pairs),
            label="Step 3/5 TIG interventional delta",
        )

        print(
            "\n"
            f"[TIG] Interventional queries: {len(valid_pairs)} "
            f"(tasks={len(tasks)}, kpis={len(kpis)}, skipped={candidate_pairs - len(valid_pairs)})"
        )

        kpi_ranges: dict[str, tuple[float, float]] = {
            kpi: (float(raw_df[kpi].min()), float(raw_df[kpi].max()))
            for kpi in kpis
            if kpi in raw_df.columns
        }

        pair_component_idx: dict[tuple[str, str], int | None] = {}
        for _, _, task, kpi in valid_pairs:
            task_comp = self.causal_analyzer.node_to_component.get(task)
            kpi_comp = self.causal_analyzer.node_to_component.get(kpi)
            pair_component_idx[(task, kpi)] = task_comp if task_comp is not None and task_comp == kpi_comp else None

        grouped_pairs: dict[tuple[str, int | None], list[tuple[int, int, str]]] = {}
        for i, j, task, kpi in valid_pairs:
            key = (task, pair_component_idx[(task, kpi)])
            grouped_pairs.setdefault(key, []).append((i, j, kpi))

        # Extract picklable args per group before entering the process pool.
        # InferenceEngine is stateful and non-picklable; we pass only the BayesianNetwork
        # so each worker constructs a fresh engine in its own process.
        group_submit_args: dict[tuple[str, int | None], tuple] = {}
        for (task, component_idx), entries in grouped_pairs.items():
            kpi_nodes = [kpi for _, _, kpi in entries]
            raw_ranges_group = {kpi: kpi_ranges[kpi] for kpi in kpi_nodes}
            if component_idx is not None and component_idx in self.causal_analyzer.component_models:
                component_nodes, bn, _ = self.causal_analyzer.component_models[component_idx]
                cols = [c for c in component_nodes if c in self.causal_analyzer.data_df.columns]
                data_records = self.causal_analyzer.data_df[cols].to_dict(orient="list")
                reps = {k: v for k, v in self.causal_analyzer.representatives.items() if k in component_nodes}
                group_submit_args[(task, component_idx)] = (
                    task, kpi_nodes, raw_ranges_group, component_nodes, bn, data_records, reps, str(output_dir) if output_dir is not None else None
                )

        t_query_start = time.perf_counter()
        _mem_snapshot("before_process_pool")
        n_workers = min(len(grouped_pairs), max_workers if max_workers is not None else (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for (task, component_idx), entries in grouped_pairs.items():
                args = group_submit_args.get((task, component_idx))
                if args is not None:
                    futures[executor.submit(_compute_effects_worker, *args)] = entries
                else:
                    # No component model — fill zeros without spawning
                    for i, j, _ in entries:
                        delta[i, j] = 0.0
                    progress.update(len(entries))

            for future in as_completed(futures):
                entries = futures[future]
                effects = future.result()
                for i, j, kpi in entries:
                    delta[i, j] = float(effects.get(kpi, 0.0))
                progress.update(len(entries))

        progress.close()
        _mem_snapshot("after_process_pool")

        self._last_delta_timing = {
            "prepare_s": t_after_prepare - t_start,
            "query_s": time.perf_counter() - t_query_start,
            "total_s": time.perf_counter() - t_start,
            "candidate_pairs": int(candidate_pairs),
            "reachable_pairs": int(len(valid_pairs)),
            "skipped_unreachable": int(candidate_pairs - len(valid_pairs)),
        }

        _mem_snapshot("delta_end")

        return delta

    def _build_action_space(self, tasks: list[str], replica_action_space: int, thread_action_space: int) -> np.ndarray:
        return np.array(
            [
                int(self.cg.nodes[task].decision_space_size)
                if task in self.cg.nodes and self.cg.nodes[task].decision_space_size > 0
                else (replica_action_space if task.endswith("_Replicas") else thread_action_space)
                for task in tasks
            ],
            dtype=float,
        )

    @staticmethod
    def _save_tig_plot(tasks: list[str], w: np.ndarray, output_png: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im = axes[0].imshow(w, cmap="viridis")
        axes[0].set_title("TIG Weight Matrix")
        axes[0].set_xticks(range(len(tasks)))
        axes[0].set_xticklabels(tasks, rotation=45, ha="right")
        axes[0].set_yticks(range(len(tasks)))
        axes[0].set_yticklabels(tasks)
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

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

    @staticmethod
    def _save_agent_assignments_yaml(partitions: dict[int, list[str]], exp_dir: Path) -> Path:
        service_to_agent: dict[str, str] = {}
        for agent_id, task_list in partitions.items():
            for task in task_list:
                service_id = task.split("_")[0]
                service_to_agent[service_id] = f"agent{agent_id}"

        assignments = {f"{service_id}.scaling": service_to_agent[service_id] for service_id in sorted(service_to_agent.keys())}

        config = {
            "control": {
                "agent_controls_replicas": True,
                "agent_controls_placement": False,
                "agent_control_assignments": assignments,
            }
        }

        agent_yaml = exp_dir / "agent_assignments.yaml"
        agent_yaml.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        return agent_yaml

    def generate_tig(
        self,
        raw_df: pd.DataFrame,
        output_dir: Path | None = None,
        dataset_path: str | None = None,
        causal_threshold: float | None = None,
        delta_mode: str = "interventional",
        bins: int = 5,
        max_parents: int = 3,
        omega: np.ndarray | None = None,
        c_max: float = 100.0,
        rho: float = 0.1,
        replica_action_space: int = 10,
        thread_action_space: int = 4,
        save_plot: bool = True,
        max_workers: int | None = None,
    ) -> TaskInterferenceGraph:
        def _mem_snapshot_local(stage: str) -> None:
            try:
                p = psutil.Process()
                rss = getattr(p.memory_info(), "rss", None)
                vms = getattr(p.memory_info(), "vms", None)
                vm = psutil.virtual_memory()
                line = f"{datetime.utcnow().isoformat()}Z,{stage},{rss},{vms},{vm.total},{vm.available}\n"
                if output_dir is not None:
                    out = Path(output_dir) / "memory_profile.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with out.open("a", encoding="utf-8") as fh:
                        fh.write(line)
            except Exception:
                pass

        _mem_snapshot_local("generate_tig_start")

        tasks = [name for name in self.variable_groups["inputs"] if name in raw_df.columns and name in self.cg.nodes]
        kpis = [name for name in self.variable_groups["outputs"] if name in raw_df.columns and name in self.cg.nodes]

        if not tasks:
            raise RuntimeError("No task variables available for TIG generation.")
        if not kpis:
            raise RuntimeError("No KPI variables available for TIG generation.")

        if delta_mode == "interventional":
            delta = self._compute_interventional_delta(
                raw_df,
                tasks,
                kpis,
                bins=bins,
                max_parents=max_parents,
                max_workers=max_workers,
                output_dir=output_dir,
            )
        else:
            delta = np.array(
                [[1.0 if kpi in self.cg.get_reachable_kpis(task) else 0.0 for kpi in kpis] for task in tasks],
                dtype=float,
            )

        _mem_snapshot_local("after_delta")

        if omega is None:
            omega_arr = np.ones(len(kpis), dtype=float)
        else:
            omega_arr = np.asarray(omega, dtype=float)
            if omega_arr.ndim != 1 or omega_arr.shape[0] != len(kpis):
                raise ValueError(f"omega must contain exactly {len(kpis)} values for the current KPI set")

        action_space = self._build_action_space(tasks, replica_action_space, thread_action_space)
        w = compute_task_interaction_weight_matrix(delta, omega_arr, action_space, c_max, rho)

        tig = TaskInterferenceGraph(
            tasks=tasks,
            kpis=kpis,
            delta=delta,
            omega=omega_arr,
            action_space=action_space,
            w=w,
            dataset_path=dataset_path,
            causal_threshold=causal_threshold,
            delta_mode=delta_mode,
            delta_bins=bins,
            delta_max_parents=max_parents,
            c_max=c_max,
            rho=rho,
            timing=dict(self._last_delta_timing),
        )
        self.last_tig = tig

        if output_dir is not None:
            self.persist_tig(tig, output_dir=output_dir, save_plot=save_plot)
            _mem_snapshot_local("after_persist_tig")

        return tig

    def persist_tig(self, tig: TaskInterferenceGraph, output_dir: Path, save_plot: bool = True) -> dict[str, Path | None]:
        output_dir.mkdir(parents=True, exist_ok=True)

        delta_csv = output_dir / "tig_delta.csv"
        pd.DataFrame(tig.delta, index=tig.tasks, columns=tig.kpis).to_csv(delta_csv)

        w_csv = output_dir / "tig_W.csv"
        pd.DataFrame(tig.w, index=tig.tasks, columns=tig.tasks).to_csv(w_csv)

        tig_edges: list[tuple[str, str, float]] = []
        for i in range(len(tig.tasks)):
            for j in range(i + 1, len(tig.tasks)):
                if tig.w[i, j] > 0:
                    tig_edges.append((tig.tasks[i], tig.tasks[j], float(tig.w[i, j])))

        edges_csv = output_dir / "tig_edges.csv"
        pd.DataFrame(tig_edges, columns=["task_i", "task_j", "weight"]).to_csv(edges_csv, index=False)

        summary_json = output_dir / "tig_summary.json"
        summary_json.write_text(
            json.dumps(
                {
                    "dataset": tig.dataset_path,
                    "causal_threshold": tig.causal_threshold,
                    "delta_mode": tig.delta_mode,
                    "delta_bins": tig.delta_bins,
                    "delta_max_parents": tig.delta_max_parents,
                    "tasks": tig.tasks,
                    "kpis": tig.kpis,
                    "task_action_space": tig.action_space.tolist(),
                    "c_max": tig.c_max,
                    "rho": tig.rho,
                    "n_tig_edges": len(tig_edges),
                    "timing": tig.timing,
                },
                indent=2,
            )
        )

        plot_png = output_dir / "tig_plot.png"
        if save_plot:
            self._save_tig_plot(tig.tasks, tig.w, plot_png)

        return {
            "delta_csv": delta_csv,
            "w_csv": w_csv,
            "edges_csv": edges_csv,
            "summary_json": summary_json,
            "plot_png": plot_png if save_plot else None,
        }

    def partition_tig(
        self,
        tig: TaskInterferenceGraph | None = None,
        w: np.ndarray | None = None,
        tasks: list[str] | None = None,
        n_agents: int = 2,
        algorithm: str = "greedy_modularity",
        seed: int | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, object]:
        if tig is None:
            if w is None and tasks is None and self.last_tig is not None:
                tig = self.last_tig
            elif w is not None and tasks is not None:
                tig = TaskInterferenceGraph(
                    tasks=list(tasks),
                    kpis=[],
                    delta=np.zeros((len(tasks), 0), dtype=float),
                    omega=np.ones((0,), dtype=float),
                    action_space=np.ones((len(tasks),), dtype=float),
                    w=np.asarray(w, dtype=float),
                )
            else:
                raise RuntimeError("Provide a TaskInterferenceGraph, or both w and tasks, or call generate_tig() first.")

        w = np.asarray(tig.w, dtype=float)
        tasks = list(tig.tasks)
        if len(tasks) == 0:
            raise RuntimeError("TIG matrix has no tasks to partition.")

        if len(tasks) <= n_agents:
            partitions = {i: [task] for i, task in enumerate(tasks)}
        elif algorithm == "spectral":
            labels = SpectralClustering(
                n_clusters=n_agents,
                affinity="precomputed",
                random_state=seed,
                assign_labels="discretize",
            ).fit_predict(w)
            partitions = {}
            for index, label in enumerate(labels):
                partitions.setdefault(int(label), []).append(tasks[index])
        else:
            ug = nx.Graph()
            ug.add_nodes_from(tasks)
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    if w[i, j] > 0:
                        ug.add_edge(tasks[i], tasks[j], weight=float(w[i, j]))

            if algorithm == "greedy_modularity":
                communities = list(nx.algorithms.community.greedy_modularity_communities(ug, weight="weight"))
                partitions = {i: sorted(list(community)) for i, community in enumerate(communities)}
            elif algorithm == "kernighan_lin":
                if n_agents != 2:
                    raise ValueError("kernighan_lin requires n_agents = 2")
                left, right = nx.algorithms.community.kernighan_lin_bisection(ug, weight="weight", seed=seed)
                partitions = {0: sorted(list(left)), 1: sorted(list(right))}
            else:
                raise ValueError(f"Unsupported partition algorithm: {algorithm}")

        result: dict[str, object] = {"partitions": partitions}

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

            partition_json = output_dir / "partitions.json"
            partition_json.write_text(json.dumps({str(k): v for k, v in partitions.items()}, indent=2))

            rows = []
            for pid, task_list in partitions.items():
                for task in task_list:
                    rows.append({"task": task, "community": int(pid)})

            partition_csv = output_dir / "partition_assignments.csv"
            pd.DataFrame(rows).sort_values(["community", "task"]).to_csv(partition_csv, index=False)

            summary_json = output_dir / "partition_summary.json"
            summary_json.write_text(
                json.dumps(
                    {
                        "algorithm": algorithm,
                        "n_agents_requested": n_agents,
                        "n_communities": len(partitions),
                        "community_sizes": {str(pid): len(ts) for pid, ts in partitions.items()},
                    },
                    indent=2,
                )
            )

            agent_yaml = self._save_agent_assignments_yaml(partitions, output_dir)
            result.update(
                {
                    "partition_json": partition_json,
                    "partition_csv": partition_csv,
                    "summary_json": summary_json,
                    "agent_yaml": agent_yaml,
                }
            )

        return result