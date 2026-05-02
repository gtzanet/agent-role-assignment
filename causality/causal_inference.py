"""Causal inference engine for observational, interventional, and counterfactual queries."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import networkx as nx
import os
from pathlib import Path
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.inference import InferenceEngine

from environment import CausalGraph


def _compute_effects_worker(
    task_node: str,
    kpi_nodes: list[str],
    raw_ranges: dict[str, tuple[float, float]],
    component_nodes: list[str],
    bn: BayesianNetwork,
    data_records: dict[str, list],
    representatives: dict[str, dict[int, float]],
    output_dir: str | None = None,
) -> dict[str, float]:
    """Module-level worker for ProcessPoolExecutor — runs in a separate OS process with its own GIL."""
    from causalnex.inference import InferenceEngine
    import pandas as pd
    import psutil
    from datetime import datetime

    def _worker_snapshot(stage: str) -> None:
        if output_dir is None:
            return
        try:
            p = psutil.Process()
            rss = getattr(p.memory_info(), "rss", None)
            vms = getattr(p.memory_info(), "vms", None)
            vm = psutil.virtual_memory()
            out = Path(output_dir) / f"memory_profile_worker_{os.getpid()}.csv"
            line = f"{datetime.utcnow().isoformat()}Z,{stage},{os.getpid()},{rss},{vms},{vm.total},{vm.available}\n"
            with out.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    ie = InferenceEngine(bn)
    data_df = pd.DataFrame(data_records).astype(int)
    _worker_snapshot("worker_start")

    valid_kpis = [kpi for kpi in kpi_nodes if kpi in component_nodes]
    if not valid_kpis or task_node not in component_nodes or task_node not in data_df.columns:
        return {kpi: 0.0 for kpi in kpi_nodes}

    task_vals = sorted(int(v) for v in data_df[task_node].dropna().unique())
    if len(task_vals) < 2:
        _worker_snapshot("worker_exit_short_task_vals")
        return {kpi: 0.0 for kpi in kpi_nodes}

    base_intervention = {v: 0.0 for v in task_vals}
    kpi_reps_map = {kpi: representatives.get(kpi, {}) for kpi in valid_kpis}
    effects_by_kpi: dict[str, list[float]] = {kpi: [] for kpi in valid_kpis}

    for x in task_vals:
        intervention = dict(base_intervention)
        intervention[x] = 1.0
        ie.do_intervention(task_node, intervention)
        q = ie.query({})
        for kpi in valid_kpis:
            dist = q[kpi]
            kpi_reps = kpi_reps_map[kpi]
            e_k = sum(float(prob) * kpi_reps.get(int(state), float(state)) for state, prob in dist.items())
            effects_by_kpi[kpi].append(e_k)
        ie.reset_do(task_node)

    _worker_snapshot("worker_after_queries")

    out: dict[str, float] = {kpi: 0.0 for kpi in kpi_nodes}
    for kpi in valid_kpis:
        raw_min, raw_max = raw_ranges.get(kpi, (0.0, 1.0))
        denom = raw_max - raw_min if (raw_max - raw_min) > 1e-12 else 1.0
        vals = effects_by_kpi[kpi]
        out[kpi] = float((max(vals) - min(vals)) / denom)

    _worker_snapshot("worker_end")

    return out


class CausalAnalyzer:
    """Encapsulates Bayesian network construction and causal queries.
    
    Initialized with a causal graph, discretizes observational data and builds
    Bayesian networks for connected components. Supports observational,
    interventional, and counterfactual queries.
    """
    
    def __init__(self, cg: CausalGraph):
        """Initialize analyzer with a causal graph.
        
        Args:
            cg: CausalGraph object containing nodes, edges, and node metadata.
        """
        self.cg = cg
        self.data_df: pd.DataFrame | None = None
        self.representatives: dict[str, dict[int, float]] | None = None
        self.component_models: dict[int, tuple[list[str], BayesianNetwork, InferenceEngine]] = {}
        self.node_to_component: dict[str, int] = {}
    
    def discretize(self, raw_df: pd.DataFrame, bins: int = 5) -> tuple[pd.DataFrame, dict[str, dict[int, float]]]:
        """Discretize continuous data for Bayesian network fitting.
        
        Args:
            raw_df: Raw continuous data.
            bins: Number of bins for quantile-based discretization.
            
        Returns:
            Tuple of (discretized_df, representatives) where representatives maps
            column names to bin values for reconstruction.
        """
        discretized = pd.DataFrame(index=raw_df.index)
        representatives = {}
        
        for col in raw_df.columns:
            s = pd.to_numeric(raw_df[col], errors="coerce")
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
            
            reps = {}
            valid = pd.DataFrame({"raw": s, "bin": cat}).dropna()
            for b in sorted(valid["bin"].unique()):
                b_int = int(b)
                reps[b_int] = float(valid[valid["bin"] == b_int]["raw"].mean())
            representatives[col] = reps
        
        self.data_df = discretized.dropna().astype(int)
        self.representatives = representatives
        return self.data_df, representatives

    def prepare_from_observations(
        self,
        raw_df: pd.DataFrame,
        columns: list[str],
        bins: int = 5,
        max_parents: int = 3,
    ) -> tuple[pd.DataFrame, dict[str, dict[int, float]], dict[int, tuple[list[str], BayesianNetwork, InferenceEngine]]]:
        """Discretize data and build component Bayesian networks in one call.

        Args:
            raw_df: Raw observational data.
            columns: Columns to use for fitting the inference model.
            bins: Number of bins for quantile discretization.
            max_parents: Maximum number of parents per node in each component BN.

        Returns:
            Tuple of (discretized_df, representatives, component_models).
        """
        selected = [c for c in columns if c in raw_df.columns]
        if not selected:
            self.data_df = pd.DataFrame()
            self.representatives = {}
            self.component_models = {}
            self.node_to_component = {}
            return self.data_df, self.representatives, self.component_models

        data_df, reps = self.discretize(raw_df[selected], bins=bins)
        component_models = self.build_network_components(list(data_df.columns), max_parents=max_parents)
        return data_df, reps, component_models
    
    def build_network_components(
        self,
        columns: list[str],
        max_parents: int = 3,
    ) -> dict[int, tuple[list[str], BayesianNetwork, InferenceEngine]]:
        """Build Bayesian networks for each connected component of the causal graph.
        
        Args:
            columns: List of columns to include in the network.
            max_parents: Maximum number of parents per node (if more exist, select by correlation).
            
        Returns:
            Dictionary mapping component_idx to (component_nodes, bayesian_network, inference_engine).
        """
        if self.data_df is None:
            raise RuntimeError("Data must be discretized first via discretize()")
        
        working_graph = self.cg.graph.subgraph(columns).copy()
        if working_graph.number_of_nodes() == 0:
            return {}
        
        components = list(nx.connected_components(working_graph.to_undirected()))
        self.component_models = {}
        self.node_to_component = {}
        
        for component_idx, component in enumerate(components):
            component_nodes = [c for c in columns if c in component]
            if len(component_nodes) < 2:
                continue
            
            component_df = self.data_df[component_nodes]
            sm = StructureModel()
            sm.add_nodes_from(component_nodes)
            
            # Build edges respecting parent limit
            parents_by_child: dict[str, list[str]] = {}
            for u, v in working_graph.edges():
                if u in component_nodes and v in component_nodes:
                    parents_by_child.setdefault(v, []).append(u)
            
            for child, parents in parents_by_child.items():
                if len(parents) <= max_parents:
                    for p in parents:
                        sm.add_edge(p, child)
                    continue
                
                # Select top-K parents by correlation
                scored = []
                for p in parents:
                    corr = component_df[p].corr(component_df[child])
                    score = abs(float(corr)) if pd.notna(corr) else 0.0
                    scored.append((score, p))
                scored.sort(reverse=True)
                for _, p in scored[:max_parents]:
                    sm.add_edge(p, child)
            
            # Fit Bayesian network
            bn = BayesianNetwork(sm)
            bn = bn.fit_node_states(component_df)
            bn = bn.fit_cpds(component_df, method="BayesianEstimator", bayes_prior="K2")
            ie = InferenceEngine(bn)
            
            self.component_models[component_idx] = (component_nodes, bn, ie)
            for node in component_nodes:
                self.node_to_component[node] = component_idx
        
        return self.component_models

    def fresh_inference_engine(self, component_idx: int) -> InferenceEngine | None:
        """Create a fresh InferenceEngine for a component, safe for concurrent use."""
        if component_idx not in self.component_models:
            return None
        _, bn, _ = self.component_models[component_idx]
        return InferenceEngine(bn)

    def interventional_effect(
        self,
        task_node: str,
        kpi_node: str,
        raw_min: float,
        raw_max: float,
        component_idx: int | None = None,
        task_vals: list[int] | None = None,
    ) -> float:
        """Compute interventional effect of a task on a KPI.
        
        Finds which component contains both nodes and uses its inference engine
        to compute the effect as: (max_kpi - min_kpi) / (raw_max - raw_min).
        
        Args:
            task_node: Name of the task/decision variable.
            kpi_node: Name of the KPI/outcome variable.
            raw_min: Minimum value of KPI in raw (pre-discretization) data.
            raw_max: Maximum value of KPI in raw data.
            
        Returns:
            Effect magnitude (0 to 1).
        """
        if self.data_df is None or self.representatives is None:
            raise RuntimeError("Data must be discretized and networks built first")
        
        if task_node not in self.data_df.columns or kpi_node not in self.data_df.columns:
            return 0.0
        
        component_nodes = None
        ie = None

        if component_idx is None:
            candidate_component = self.node_to_component.get(task_node)
            if candidate_component is not None and self.node_to_component.get(kpi_node) == candidate_component:
                component_idx = candidate_component

        if component_idx is not None and component_idx in self.component_models:
            nodes, _, inference_engine = self.component_models[component_idx]
            if task_node in nodes and kpi_node in nodes:
                component_nodes = nodes
                ie = inference_engine

        if ie is None:
            for idx, (nodes, _, inference_engine) in self.component_models.items():
                if task_node in nodes and kpi_node in nodes:
                    component_idx = idx
                    component_nodes = nodes
                    ie = inference_engine
                    break
        
        if ie is None:
            return 0.0
        
        if task_vals is None:
            component_df = self.data_df[component_nodes]
            task_vals = sorted(component_df[task_node].dropna().unique().tolist())
        if len(task_vals) < 2:
            return 0.0
        
        task_vals_int = [int(v) for v in task_vals]
        base_intervention = {v: 0.0 for v in task_vals_int}
        kpi_reps = self.representatives.get(kpi_node, {})

        # Compute effect across interventions
        effects = [0.0] * len(task_vals_int)
        for idx, x in enumerate(task_vals_int):
            intervention = dict(base_intervention)
            intervention[x] = 1.0
            ie.do_intervention(task_node, intervention)
            
            q = ie.query({})
            dist = q[kpi_node]
            
            e_k = 0.0
            for state, prob in dist.items():
                state_int = int(state)
                rep = kpi_reps.get(state_int, float(state_int))
                e_k += float(prob) * rep
            effects[idx] = e_k
            ie.reset_do(task_node)
        
        denom = raw_max - raw_min
        if denom <= 1e-12:
            denom = 1.0
        
        return float((max(effects) - min(effects)) / denom)

    def interventional_effects_for_task(
        self,
        task_node: str,
        kpi_nodes: list[str],
        raw_ranges: dict[str, tuple[float, float]],
        component_idx: int | None = None,
        inference_engine: InferenceEngine | None = None,
    ) -> dict[str, float]:
        """Compute interventional effects for one task against multiple KPIs in one pass."""
        if self.data_df is None or self.representatives is None:
            raise RuntimeError("Data must be discretized and networks built first")

        if task_node not in self.data_df.columns or not kpi_nodes:
            return {kpi: 0.0 for kpi in kpi_nodes}

        if component_idx is None:
            component_idx = self.node_to_component.get(task_node)

        if component_idx is None or component_idx not in self.component_models:
            return {kpi: 0.0 for kpi in kpi_nodes}

        component_nodes, _, stored_ie = self.component_models[component_idx]
        ie = inference_engine if inference_engine is not None else stored_ie
        if task_node not in component_nodes:
            return {kpi: 0.0 for kpi in kpi_nodes}

        valid_kpis = [
            kpi
            for kpi in kpi_nodes
            if kpi in component_nodes and self.node_to_component.get(kpi) == component_idx
        ]
        if not valid_kpis:
            return {kpi: 0.0 for kpi in kpi_nodes}

        component_df = self.data_df[component_nodes]
        task_vals = sorted(component_df[task_node].dropna().unique().tolist())
        if len(task_vals) < 2:
            return {kpi: 0.0 for kpi in kpi_nodes}

        task_vals_int = [int(v) for v in task_vals]
        base_intervention = {v: 0.0 for v in task_vals_int}
        kpi_reps_map = {kpi: self.representatives.get(kpi, {}) for kpi in valid_kpis}
        effects_by_kpi: dict[str, list[float]] = {kpi: [0.0] * len(task_vals_int) for kpi in valid_kpis}
        for idx, x in enumerate(task_vals_int):
            intervention = dict(base_intervention)
            intervention[x] = 1.0
            ie.do_intervention(task_node, intervention)

            q = ie.query({})
            for kpi in valid_kpis:
                dist = q[kpi]
                kpi_reps = kpi_reps_map[kpi]
                e_k = 0.0
                for state, prob in dist.items():
                    state_int = int(state)
                    rep = kpi_reps.get(state_int, float(state_int))
                    e_k += float(prob) * rep
                effects_by_kpi[kpi][idx] = e_k

            ie.reset_do(task_node)

        out: dict[str, float] = {kpi: 0.0 for kpi in kpi_nodes}
        for kpi in valid_kpis:
            raw_min, raw_max = raw_ranges.get(kpi, (0.0, 1.0))
            denom = raw_max - raw_min
            if denom <= 1e-12:
                denom = 1.0
            out[kpi] = float((max(effects_by_kpi[kpi]) - min(effects_by_kpi[kpi])) / denom)

        return out
    
    def observational_query(self, evidence: dict[str, int] | None = None) -> dict[str, dict]:
        """Perform observational query on the networks.
        
        Args:
            evidence: Dictionary of variable assignments for conditional queries.
                     If None, returns unconditional distributions.
                     
        Returns:
            Dictionary mapping variable names to their probability distributions.
        """
        if not self.component_models:
            raise RuntimeError("Networks must be built first via build_network_components()")
        
        results = {}
        evidence = evidence or {}
        
        for idx, (nodes, bn, ie) in self.component_models.items():
            component_evidence = {k: v for k, v in evidence.items() if k in nodes}
            distributions = ie.query(component_evidence)
            results.update(distributions)
        
        return results
    
    def counterfactual_query(
        self,
        evidence: dict[str, int],
        intervention: dict[str, int],
    ) -> dict[str, dict]:
        """Perform counterfactual query: what if intervention happened given observed evidence?
        
        Args:
            evidence: Observed variable assignments.
            intervention: Hypothetical intervention to apply.
            
        Returns:
            Dictionary of counterfactual distributions.
        """
        if not self.component_models:
            raise RuntimeError("Networks must be built first via build_network_components()")
        
        results = {}
        
        for idx, (nodes, bn, ie) in self.component_models.items():
            component_evidence = {k: v for k, v in evidence.items() if k in nodes}
            component_intervention = {k: v for k, v in intervention.items() if k in nodes}
            
            if component_intervention:
                for var, val in component_intervention.items():
                    ie.do_intervention(var, {int(val): 1.0})
            
            distributions = ie.query(component_evidence)
            results.update(distributions)
            
            # Reset interventions
            for var in component_intervention:
                ie.reset_do(var)

        return results


class CausalAnalyzerWrapper:
    """HTTP proxy for a CausalAnalyzer running on a remote Flask server.

    Implements the same public interface as CausalAnalyzer so it can be used
    as a drop-in replacement. The BayesianNetwork lives only on the server,
    eliminating the cost of duplicating it across local OS processes.

    The server is expected to be fully initialised (causal graph + BN built)
    before the wrapper is used. cg is accepted for interface compatibility but
    is not sent to the server — the server builds its own graph at startup.

    Usage:
        wrapper = CausalAnalyzerWrapper(cg, base_url="http://localhost:5050")
        wrapper.prepare_from_observations(raw_df, columns)
        effects = wrapper.interventional_effects_for_task(task, kpis, raw_ranges)
    """

    def __init__(self, cg: CausalGraph, base_url: str):  # cg accepted for interface compatibility
        del cg
        import requests as _req
        self._requests = _req
        self._base_url = base_url.rstrip("/")
        self.data_df: pd.DataFrame | None = None
        self.representatives: dict[str, dict[int, float]] | None = None
        self.node_to_component: dict[str, int] = {}
        # component_models holds (nodes, None, None) — BN lives on the server
        self.component_models: dict[int, tuple[list[str], None, None]] = {}

    def _post(self, endpoint: str, payload: dict) -> dict:
        resp = self._requests.post(f"{self._base_url}/{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _deserialize_df(payload: str) -> pd.DataFrame:
        return pd.read_json(io.StringIO(payload), orient="split")

    def prepare_from_observations(
        self,
        raw_df: pd.DataFrame,  # noqa: ARG002 — server owns the data
        columns: list[str],
        bins: int = 5,          # noqa: ARG002 — fixed at server startup
        max_parents: int = 3,   # noqa: ARG002 — fixed at server startup
    ) -> tuple[pd.DataFrame, dict[str, dict[int, float]], dict]:
        # raw_df, bins, and max_parents are owned by the server (fixed at startup).
        # Only columns is sent so the server can filter its pre-built state.
        del raw_df, bins, max_parents  # accepted for interface compatibility only
        result = self._post("prepare_from_observations", {"columns": columns})
        self.data_df = self._deserialize_df(result["data_df"])
        self.representatives = {
            col: {int(k): float(v) for k, v in bins_map.items()}
            for col, bins_map in result["representatives"].items()
        }
        self.node_to_component = {k: int(v) for k, v in result["node_to_component"].items()}
        self.component_models = {
            int(idx): (nodes, None, None)
            for idx, nodes in result["component_nodes_by_idx"].items()
        }
        return self.data_df, self.representatives, self.component_models

    def interventional_effects_for_task(
        self,
        task_node: str,
        kpi_nodes: list[str],
        raw_ranges: dict[str, tuple[float, float]],
        component_idx: int | None = None,
        _inference_engine=None,  # ignored — kept for interface compatibility
    ) -> dict[str, float]:
        result = self._post("interventional_effects_for_task", {
            "task_node": task_node,
            "kpi_nodes": kpi_nodes,
            "raw_ranges": {k: list(v) for k, v in raw_ranges.items()},
            "component_idx": component_idx,
        })
        return result["effects"]

    def interventional_effect(
        self,
        task_node: str,
        kpi_node: str,
        raw_min: float,
        raw_max: float,
        component_idx: int | None = None,
        task_vals: list[int] | None = None,
    ) -> float:
        result = self._post("interventional_effect", {
            "task_node": task_node,
            "kpi_node": kpi_node,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "component_idx": component_idx,
            "task_vals": task_vals,
        })
        return float(result["effect"])

    def observational_query(self, evidence: dict[str, int] | None = None) -> dict[str, dict]:
        result = self._post("observational_query", {"evidence": evidence})
        return result["distributions"]

    def counterfactual_query(
        self,
        evidence: dict[str, int],
        intervention: dict[str, int],
    ) -> dict[str, dict]:
        result = self._post("counterfactual_query", {
            "evidence": evidence,
            "intervention": intervention,
        })
        return result["distributions"]