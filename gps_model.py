"""
GPS system model loaded from an analytical config dict.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class GPSModel:
    """Analytical GPS system model loaded from a config dict."""

    def __init__(self, cfg: dict):
        self.N = cfg["system"]["N"]
        self.M = cfg["system"]["M"]
        self.L = cfg["system"]["L"]
        # NOTE: no longer changes any GPSModel computation. Every service's
        # egress is always capacity-bounded (_lam_in_chain) regardless of
        # this flag -- both D_wf (drop) and sigma_w/latency (queue) are
        # computed unconditionally from that same egress-capped chain. This
        # is kept as a documentary label for which interpretation (discarded
        # vs. queued) applies to a scenario; u_metric/equilibrium_metric are
        # what actually select which KPI/utility gets reported.
        self.drop_on_overload = cfg["system"].get("drop_on_overload", True)
        self.latency_threshold_pct = cfg["system"].get("latency_threshold_pct", 200)
        self.u_metric = cfg["system"].get("u_metric", "utilization")
        if self.u_metric not in ("utilization", "latency_success"):
            raise ValueError(
                f"system.u_metric must be 'utilization' or 'latency_success', got {self.u_metric!r}"
            )
        self.delta = cfg["delta"]
        self.f = {n["id"]: n["f"] for n in cfg["nodes"]}

        self.services: dict = {}
        self.workflows: dict = {}
        for wf in cfg["workflows"]:
            wid = wf["id"]
            self.workflows[wid] = {"lambda": wf["lambda"], "services": []}
            for stage, svc in enumerate(wf["services"]):
                sid = svc["id"]
                self.services[sid] = {
                    "node": svc["node"],
                    "workflow": wid,
                    "stage": stage,
                    "r_base": svc["replicas_baseline"],
                    "A": svc["action_space"],
                }
                self.workflows[wid]["services"].append(sid)

        self.node_services: dict = {n: [] for n in range(self.N)}
        for sid, svc in self.services.items():
            self.node_services[svc["node"]].append(sid)

        w_drop = cfg["tig"]["kpi_weights"]["drop_rate"]
        w_util = cfg["tig"]["kpi_weights"]["utilization"]
        self.kpi_names = ([f"D_wf{w}/λ" for w in range(self.M)]
                          + [f"u_node{n}" for n in range(self.N)])
        self.kpi_weights = [w_drop / self.M] * self.M + [w_util / self.N] * self.N
        self.omega = np.array(self.kpi_weights)
        self.task_ids = sorted(self.services.keys())

    def _lam_in_chain(self, r: dict, C: dict, lambda_obs: Optional[dict] = None) -> dict:
        """Cascading arrival rate reaching each service: lam_in[s] = lambda_w
        for the first service in a workflow's chain, min(lam_in[s'], C[s'])
        of the previous service s' for every subsequent one. A service can
        never receive more than what the previous stage could actually
        throughput -- this is a physical egress constraint that holds
        regardless of drop_on_overload. That flag only governs what happens
        to the resulting gap (lam_in - egress): discarded (contributes to
        D_wf) when the reliability mechanism is engaged, or queued/delayed
        (contributes to latency, see _workflow_latency_success) when it is
        not -- either way, downstream services only ever see the
        capacity-bounded egress, never the raw upstream demand.
        """
        lam_in = {}
        for wid, wf in self.workflows.items():
            lam_w = lambda_obs[wid] if lambda_obs is not None else wf["lambda"]
            lam_cur = lam_w
            for sid in wf["services"]:
                lam_in[sid] = lam_cur
                lam_cur = min(lam_cur, C[sid])
        return lam_in

    def compute_kpis(self, r: dict) -> dict:
        """GPS KPIs for replica assignment r = {service_id: replicas}."""
        R = {n: sum(r[s] for s in sids) for n, sids in self.node_services.items()}

        C = {}
        for sid, svc in self.services.items():
            n = svc["node"]
            R_n = R[n]
            eff_f = self.f[n] - self.delta * R_n
            C[sid] = r[sid] * eff_f / (self.L * R_n) if eff_f > 0 else 0.0

        kpis = {}

        # D_wf: structural drop rate implied by the bottleneck capacity vs.
        # offered load -- a property of the topology/assignment, not of
        # whether a mechanism actually enforces it.
        lam_in = self._lam_in_chain(r, C)
        for wid, wf in self.workflows.items():
            lam_w = wf["lambda"]
            bottleneck = min(C[sid] for sid in wf["services"])
            kpis[f"D_wf{wid}/λ"] = max(lam_w - bottleneck, 0.0) / lam_w * 100.0

        # Utilization: admitted flow / capacity. u_metric picks whether this
        # or latency_success below feeds the scored u_node{n} KPI.
        util = {}
        for n in range(self.N):
            R_n = R[n]
            eff_f = self.f[n] - self.delta * R_n
            if eff_f <= 0 or R_n == 0:
                util[n] = 100.0
            else:
                admitted = sum(min(lam_in[s], C[s]) * self.L
                               for s in self.node_services[n])
                util[n] = admitted / eff_f * 100.0

        # Latency-SLA success %: also computed regardless of drop_on_overload.
        success = self._latency_success(r, R, C)

        for n in range(self.N):
            kpis[f"u_node{n}"] = util[n] if self.u_metric == "utilization" else success[n]
            kpis[f"u_node{n}_util"] = util[n]
            kpis[f"u_node{n}_success"] = success[n]

        return kpis

    def workflow_latency_success(self, r: dict) -> dict:
        """Per-workflow binary latency-SLA success indicator sigma_w in
        {0.0, 1.0}: whether the workflow's end-to-end latency is <=
        latency_threshold_pct% of the zero-contention baseline (sum of
        L/eff_f over its service chain). Excess demand above a stage's
        capacity is not discarded here (it queues and is penalised via
        latency, see below) but each stage still only ever *receives* the
        egress-capped throughput of the previous stage (_lam_in_chain) --
        this differs from drop_on_overload=True only in what happens to the
        gap, never in how much traffic physically reaches a given stage.

        Per-service latency E[T_s] uses an M/D/1 approximation with a
        dynamic rate mu_s* = mu_s + idle_node_capacity * (r_s / R_n): each
        service keeps its guaranteed share C[sid], plus its
        replica-proportional cut of whatever node capacity co-located
        services aren't using.
        """
        R = {n: sum(r[s] for s in sids) for n, sids in self.node_services.items()}
        C = {}
        for sid, svc in self.services.items():
            n = svc["node"]
            R_n = R[n]
            eff_f = self.f[n] - self.delta * R_n
            C[sid] = r[sid] * eff_f / (self.L * R_n) if eff_f > 0 else 0.0
        return self._workflow_latency_success(r, R, C)

    def _workflow_latency_success(self, r: dict, R: dict, C: dict) -> dict:
        eff_f = {}
        mu_node = {}
        for n in range(self.N):
            f_n = self.f[n] - self.delta * R[n]
            eff_f[n] = f_n
            mu_node[n] = f_n / self.L if f_n > 0 else 0.0

        lam_in = self._lam_in_chain(r, C)

        E_T: dict[int, float] = {}
        for sid, svc in self.services.items():
            n, R_n = svc["node"], R[svc["node"]]
            if mu_node[n] <= 0 or R_n == 0:
                E_T[sid] = float("inf")
                continue
            lam_other = sum(lam_in[s2] for s2 in self.node_services[n] if s2 != sid)
            rho_other = lam_other / mu_node[n]
            mu_star = C[sid] + mu_node[n] * max(1.0 - rho_other, 0.0) * (r[sid] / R_n)
            if mu_star <= lam_in[sid]:
                E_T[sid] = float("inf")
            else:
                E_T[sid] = (1.0 / mu_star
                            + lam_in[sid] / (2.0 * mu_star * (mu_star - lam_in[sid])))

        wf_success = {}
        for wid, wf in self.workflows.items():
            t_wf = sum(E_T[sid] for sid in wf["services"])
            threshold = sum(self.L / eff_f[self.services[sid]["node"]]
                             for sid in wf["services"]
                             if eff_f[self.services[sid]["node"]] > 0)
            threshold *= self.latency_threshold_pct / 100.0
            wf_success[wid] = 1.0 if t_wf <= threshold else 0.0
        return wf_success

    def _latency_success(self, r: dict, R: dict, C: dict) -> dict:
        """% of requests through node n whose workflow met its latency SLA
        (see workflow_latency_success), weighted by each service's share of
        the node's actual (egress-capped) incoming traffic.
        """
        wf_success = self._workflow_latency_success(r, R, C)
        lam_in = self._lam_in_chain(r, C)

        success = {}
        for n in range(self.N):
            sids = self.node_services[n]
            total = sum(lam_in[s] for s in sids)
            if total <= 0:
                success[n] = 0.0
            else:
                success_flow = sum(lam_in[s] * wf_success[self.services[s]["workflow"]]
                                    for s in sids)
                success[n] = success_flow / total * 100.0
        return success

    def score(self, kpis: dict) -> float:
        """Weighted composite score J (lower = better; 0 = perfect)."""
        return float(np.dot(self.omega, [kpis[k] for k in self.kpi_names]))

    def compute_service_stats(self, r: dict, lambda_obs: Optional[dict] = None) -> dict:
        """
        Per-service capacity, effective arrival rate, utilisation, and
        per-workflow ergodicity ratio.

        lambda_obs: optional {wid: observed_lambda} override used for rho_wf
          and lam_eff. When None the true workflow lambdas from config are used.
          Pass a noisy observation to simulate imperfect lambda measurement
          while keeping compute_kpis() working with the true lambdas.

        Returns:
          C         {sid: capacity (req/s)}
          lam_eff   {sid: effective arrival rate (req/s)}
          rho_s     {sid: lam_eff_s / C_s}  — per-service utilisation
          rho_wf    {wid: lambda_w / C_{last_service_w}}  — ergodicity metric
        """
        R = {n: sum(r[s] for s in sids) for n, sids in self.node_services.items()}

        C = {}
        for sid, svc in self.services.items():
            n = svc["node"]
            R_n = R[n]
            eff_f = self.f[n] - self.delta * R_n
            C[sid] = r[sid] * eff_f / (self.L * R_n) if eff_f > 0 else 0.0

        lam_in = self._lam_in_chain(r, C, lambda_obs)
        lam_eff = {sid: min(lam_in[sid], C[sid]) for sid in self.services}

        rho_s = {
            sid: (lam_eff[sid] / C[sid] if C[sid] > 0 else 0.0)
            for sid in self.services
        }

        rho_wf = {
            wid: ((lambda_obs[wid] if lambda_obs is not None else wf["lambda"])
                  / min(C[s] for s in wf["services"])
                  if min(C[s] for s in wf["services"]) > 0 else float("inf"))
            for wid, wf in self.workflows.items()
        }

        return {"C": C, "lam_in": lam_in, "lam_eff": lam_eff, "rho_s": rho_s, "rho_wf": rho_wf}

    @property
    def baseline_assignment(self) -> dict:
        return {sid: svc["r_base"] for sid, svc in self.services.items()}