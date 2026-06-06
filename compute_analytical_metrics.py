"""
GPS system metrics module.

Importable:
    model = GPSModel(cfg)          # cfg: parsed YAML dict
    kpis  = model.compute_kpis(r)  # r: {service_id: replicas}
    J     = model.score(kpis)

CLI (--config analytical_config.yaml):
    Evaluates baseline, optimal, and worst-case operating points and writes
    results under results/analytical/<scenario>/metrics/.
"""

import argparse
import json
from itertools import product as cart_product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

MAX_SWEEP_ROWS = 500_000


class GPSModel:
    """Analytical GPS system model loaded from a config dict."""

    def __init__(self, cfg: dict):
        self.N     = cfg["system"]["N"]
        self.M     = cfg["system"]["M"]
        self.L     = cfg["system"]["L"]
        self.delta = cfg["delta"]
        self.f     = {n["id"]: n["f"] for n in cfg["nodes"]}

        self.services:  dict = {}
        self.workflows: dict = {}
        for wf in cfg["workflows"]:
            wid = wf["id"]
            self.workflows[wid] = {"lambda": wf["lambda"], "services": []}
            for stage, svc in enumerate(wf["services"]):
                sid = svc["id"]
                self.services[sid] = {
                    "node":     svc["node"],
                    "workflow": wid,
                    "stage":    stage,
                    "r_base":   svc["replicas_baseline"],
                    "A":        svc["action_space"],
                }
                self.workflows[wid]["services"].append(sid)

        self.node_services: dict = {n: [] for n in range(self.N)}
        for sid, svc in self.services.items():
            self.node_services[svc["node"]].append(sid)

        w_drop = cfg["tig"]["kpi_weights"]["drop_rate"]
        w_util = cfg["tig"]["kpi_weights"]["utilization"]
        self.kpi_names   = ([f"D_wf{w}/λ" for w in range(self.M)]
                            + [f"u_node{n}" for n in range(self.N)])
        self.kpi_weights = [w_drop / self.M] * self.M + [w_util / self.N] * self.N
        self.omega       = np.array(self.kpi_weights)
        self.task_ids    = sorted(self.services.keys())

    # ── Core KPI computation ──────────────────────────────────────────────────

    def compute_kpis(self, r: dict) -> dict:
        """GPS KPIs for replica assignment r = {service_id: replicas}."""
        R = {n: sum(r[s] for s in sids) for n, sids in self.node_services.items()}

        C = {}
        for sid, svc in self.services.items():
            n     = svc["node"]
            R_n   = R[n]
            eff_f = self.f[n] - self.delta * R_n
            C[sid] = r[sid] * eff_f / (self.L * R_n) if eff_f > 0 else 0.0

        kpis    = {}
        lam_eff = {}
        for wid, wf in self.workflows.items():
            lam_w   = wf["lambda"]
            lam_cur = lam_w
            for sid in wf["services"]:
                lam_eff[sid] = lam_cur
                lam_cur      = min(lam_cur, C[sid])
            bottleneck       = min(C[sid] for sid in wf["services"])
            kpis[f"D_wf{wid}/λ"] = max(lam_w - bottleneck, 0.0) / lam_w * 100.0

        for n in range(self.N):
            R_n   = R[n]
            eff_f = self.f[n] - self.delta * R_n
            if eff_f <= 0 or R_n == 0:
                kpis[f"u_node{n}"] = 100.0
            else:
                admitted         = sum(min(lam_eff[s], C[s]) * self.L
                                       for s in self.node_services[n])
                kpis[f"u_node{n}"] = admitted / eff_f * 100.0

        return kpis

    def score(self, kpis: dict) -> float:
        """Weighted composite score J (lower = better; 0 = perfect)."""
        return float(np.dot(self.omega, [kpis[k] for k in self.kpi_names]))

    def compute_service_stats(self, r: dict,
                              lambda_obs: Optional[dict] = None) -> dict:
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
            n     = svc["node"]
            R_n   = R[n]
            eff_f = self.f[n] - self.delta * R_n
            C[sid] = r[sid] * eff_f / (self.L * R_n) if eff_f > 0 else 0.0

        lam_eff = {}
        for wid, wf in self.workflows.items():
            lam_w   = lambda_obs[wid] if lambda_obs is not None else wf["lambda"]
            lam_cur = lam_w
            for sid in wf["services"]:
                lam_eff[sid] = lam_cur
                lam_cur      = min(lam_cur, C[sid])

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


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="analytical_config.yaml")
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config)
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    SCENARIO = cfg.get("name", CONFIG_PATH.stem)
    model    = GPSModel(cfg)

    print(f"Config   : {CONFIG_PATH}")
    print(f"Scenario : {SCENARIO}")
    print(f"Nodes={model.N}  Workflows={model.M}  L={model.L}  δ={model.delta}")
    print(f"KPIs     : {model.kpi_names}")
    print(f"Weights  : {[round(w, 4) for w in model.kpi_weights]}")

    # ── Full grid sweep ───────────────────────────────────────────────────────
    action_spaces = [model.services[s]["A"] for s in model.task_ids]
    n_combos      = 1
    for A in action_spaces:
        n_combos *= len(A)

    print(f"\nAction space: {'×'.join(str(len(A)) for A in action_spaces)} = {n_combos:,} combinations")

    sweep_df       = None
    best_r         = None
    worst_r        = None

    if n_combos <= MAX_SWEEP_ROWS:
        print("Sweeping full grid...")
        records     = []
        best_score  = np.inf
        worst_score = -np.inf

        for r_tuple in cart_product(*action_spaces):
            r    = dict(zip(model.task_ids, r_tuple))
            kpis = model.compute_kpis(r)
            J    = model.score(kpis)
            row  = dict(zip(model.task_ids, r_tuple))
            row.update(kpis)
            row["J"] = J
            records.append(row)
            if J < best_score:
                best_score = J
                best_r     = r_tuple
            if J > worst_score:
                worst_score = J
                worst_r     = r_tuple

        sweep_df = pd.DataFrame(records)
        print(f"Sweep done.  best J={best_score:.3f}  worst J={worst_score:.3f}")
    else:
        print(f"Grid too large ({n_combos:,} > {MAX_SWEEP_ROWS:,}) — baseline only.")

    # ── Operating points ──────────────────────────────────────────────────────
    r_base    = model.baseline_assignment
    kpis_base = model.compute_kpis(r_base)
    J_base    = model.score(kpis_base)

    operating_points = {"baseline": (r_base, kpis_base, J_base)}

    if best_r is not None:
        r_opt      = dict(zip(model.task_ids, best_r))
        kpis_opt   = model.compute_kpis(r_opt)
        J_opt      = model.score(kpis_opt)
        r_worst    = dict(zip(model.task_ids, worst_r))
        kpis_worst = model.compute_kpis(r_worst)
        J_worst    = model.score(kpis_worst)
        operating_points["optimal"] = (r_opt,   kpis_opt,   J_opt)
        operating_points["worst"]   = (r_worst, kpis_worst, J_worst)

    # ── Console summary ───────────────────────────────────────────────────────
    for label, (r, kpis, J) in operating_points.items():
        print(f"\n{'─'*60}")
        print(f"  {label.upper()}")
        print(f"  Assignment: { {f's{k}': v for k, v in r.items()} }")
        print(f"  J = {J:.3f}")
        for k in model.kpi_names:
            print(f"    {k:<15} = {kpis[k]:6.2f} %")

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUT_DIR = Path("results/analytical") / SCENARIO / "metrics"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, (r, kpis, J) in operating_points.items():
        assign_str = str({f"s{s}": r[s] for s in model.task_ids})
        rows = [{"metric": k, "value": round(kpis[k], 4), "unit": "%",
                 "assignment": assign_str}
                for k in model.kpi_names]
        rows.append({"metric": "J", "value": round(J, 4), "unit": "",
                     "assignment": assign_str})
        pd.DataFrame(rows).to_csv(OUT_DIR / f"metrics_{label}.csv", index=False)

    if sweep_df is not None:
        sweep_df.to_csv(OUT_DIR / "sweep.csv", index=False)
        print(f"\nSaved sweep.csv  ({len(sweep_df):,} rows)")

    summary = {
        "scenario":  SCENARIO,
        "N": model.N, "M": model.M, "L": model.L, "delta": model.delta,
        "kpi_names":   model.kpi_names,
        "kpi_weights": list(model.omega),
        "operating_points": {
            label: {
                "assignment": {f"s{s}": int(r[s]) for s in model.task_ids},
                "kpis":       {k: round(v, 4) for k, v in kpis.items()},
                "J":          round(J, 4),
            }
            for label, (r, kpis, J) in operating_points.items()
        },
        "sweep_rows": len(sweep_df) if sweep_df is not None else None,
    }
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nOutputs → {OUT_DIR}/")
    for f_out in sorted(OUT_DIR.iterdir()):
        print(f"  {f_out.name}")
