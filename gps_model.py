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
        lam_eff = {}
        for wid, wf in self.workflows.items():
            lam_w = wf["lambda"]
            lam_cur = lam_w
            for sid in wf["services"]:
                lam_eff[sid] = lam_cur
                lam_cur = min(lam_cur, C[sid])
            bottleneck = min(C[sid] for sid in wf["services"])
            kpis[f"D_wf{wid}/λ"] = max(lam_w - bottleneck, 0.0) / lam_w * 100.0

        for n in range(self.N):
            R_n = R[n]
            eff_f = self.f[n] - self.delta * R_n
            if eff_f <= 0 or R_n == 0:
                kpis[f"u_node{n}"] = 100.0
            else:
                admitted = sum(min(lam_eff[s], C[s]) * self.L
                               for s in self.node_services[n])
                kpis[f"u_node{n}"] = admitted / eff_f * 100.0

        return kpis

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

        lam_eff = {}
        for wid, wf in self.workflows.items():
            lam_w = lambda_obs[wid] if lambda_obs is not None else wf["lambda"]
            lam_cur = lam_w
            for sid in wf["services"]:
                lam_eff[sid] = lam_cur
                lam_cur = min(lam_cur, C[sid])

        rho_s = {
            sid: (lam_eff[sid] / C[sid] if C[sid] > 0 else float("inf"))
            for sid in self.services
        }

        rho_wf = {
            wid: ((lambda_obs[wid] if lambda_obs is not None else wf["lambda"])
                  / C[wf["services"][-1]]
                  if C[wf["services"][-1]] > 0 else float("inf"))
            for wid, wf in self.workflows.items()
        }

        return {"C": C, "lam_eff": lam_eff, "rho_s": rho_s, "rho_wf": rho_wf}

    @property
    def baseline_assignment(self) -> dict:
        return {sid: svc["r_base"] for sid, svc in self.services.items()}