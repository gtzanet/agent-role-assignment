"""
Compute TIG edge weights for all four delta methods from the analytical GPS model.

Outputs under results/analytical/TIG/<scenario>/<method>/:
  W.csv          — final TIG edge weight matrix W_{i,j}
  F_pull.csv     — normalised pull-force matrix
  F_push.csv     — complexity-penalty matrix (same for all methods)
  indiv.csv      — per-task individual sensitivity v({i}, k)
  config.yaml    — copy of the config used
  meta.json      — run metadata

The scenario name is read from the 'name' field in the config; falls back to the
config filename stem if absent.
"""

import argparse
import json
import shutil
from itertools import combinations as icomb
from itertools import product as cart_product
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="analytical_config.yaml")
args = parser.parse_args()

CONFIG_PATH = Path(args.config)


# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

SCENARIO = cfg.get("name", CONFIG_PATH.stem)

N   = cfg["system"]["N"]
M   = cfg["system"]["M"]
L   = cfg["system"]["L"]
dlt = cfg["delta"]

f = {n["id"]: n["f"] for n in cfg["nodes"]}

services:  dict = {}
workflows: dict = {}
for wf in cfg["workflows"]:
    wid = wf["id"]
    workflows[wid] = {"lambda": wf["lambda"], "services": []}
    for stage, svc in enumerate(wf["services"]):
        sid = svc["id"]
        services[sid] = {
            "node":     svc["node"],
            "workflow": wid,
            "stage":    stage,
            "r_base":   svc["replicas_baseline"],
            "A":        svc["action_space"],
        }
        workflows[wid]["services"].append(sid)

node_services: dict = {n: [] for n in range(N)}
for sid, svc in services.items():
    node_services[svc["node"]].append(sid)

C_max  = cfg["tig"]["C_max"]
rho    = cfg["tig"]["rho"]
w_drop = cfg["tig"]["kpi_weights"]["drop_rate"]
w_util = cfg["tig"]["kpi_weights"]["utilization"]

kpi_names   = [f"D_wf{w}/λ" for w in range(M)] + [f"u_node{n}" for n in range(N)]
kpi_weights = [w_drop / M] * M + [w_util / N] * N
omega       = np.array(kpi_weights)

task_ids   = sorted(services.keys())
task_names = [f"t{i}(s{i})" for i in task_ids]
n_tasks    = len(task_ids)
n_kpis     = len(kpi_names)

METHODS = ["cosine", "sii_marginalised", "sii_null", "sobol"]

print(f"Config  : {CONFIG_PATH}")
print(f"Scenario: {SCENARIO}")
print(f"Nodes={N}  Workflows={M}  L={L}  δ={dlt}")
print(f"Tasks   : {task_names}")
print(f"KPIs    : {kpi_names}")
print(f"C_max={C_max}  ρ={rho}")


# ── GPS KPI functions ─────────────────────────────────────────────────────────
def compute_R(r):
    return {n: sum(r[s] for s in sids) for n, sids in node_services.items()}


def compute_C(r, R):
    C = {}
    for sid, svc in services.items():
        n     = svc["node"]
        R_n   = R[n]
        eff_f = f[n] - dlt * R_n
        C[sid] = r[sid] * eff_f / (L * R_n) if eff_f > 0 else 0.0
    return C


def propagate_workflow(wid, C):
    wf      = workflows[wid]
    lam_w   = wf["lambda"]
    chain   = wf["services"]
    lam_eff = {}
    lam_cur = lam_w
    for sid in chain:
        lam_eff[sid] = lam_cur
        lam_cur      = min(lam_cur, C[sid])
    bottleneck = min(C[sid] for sid in chain)
    D_w        = max(lam_w - bottleneck, 0.0)
    return lam_eff, (D_w / lam_w) * 100.0


def compute_kpis(r):
    R = compute_R(r)
    C = compute_C(r, R)
    kpis    = {}
    lam_eff = {}
    for wid in range(M):
        eff, d_norm = propagate_workflow(wid, C)
        lam_eff.update(eff)
        kpis[f"D_wf{wid}/λ"] = d_norm
    for n in range(N):
        R_n   = R[n]
        eff_f = f[n] - dlt * R_n
        if eff_f <= 0 or R_n == 0:
            kpis[f"u_node{n}"] = 100.0
        else:
            admitted = sum(min(lam_eff[s], C[s]) * L for s in node_services[n])
            kpis[f"u_node{n}"] = (admitted / eff_f) * 100.0
    return kpis


# ── Precompute total KPI variance for Sobol ───────────────────────────────────
print("\nPrecomputing full KPI grid for Sobol total variance...")
_all_kpi_grid = {
    k: [compute_kpis(dict(zip(task_ids, r)))[k]
        for r in cart_product(*[services[s]["A"] for s in task_ids])]
    for k in kpi_names
}
_total_var = {k: np.var(_all_kpi_grid[k]) for k in kpi_names}


# ── Value function v(S, k, method) ───────────────────────────────────────────
def v_func(S, k, method):
    S      = list(S)
    others = [s for s in task_ids if s not in S]
    A_S    = [services[s]["A"] for s in S]
    A_oth  = [services[s]["A"] for s in others]
    if not S:
        return 0.0

    if method in ("cosine", "sii_marginalised"):
        total, n_cfgs = 0.0, 0
        for r_oth in (cart_product(*A_oth) if others else [()]):
            r_fix  = dict(zip(others, r_oth))
            vals   = [compute_kpis({**r_fix, **dict(zip(S, rS))})[k]
                      for rS in cart_product(*A_S)]
            total  += max(vals) - min(vals)
            n_cfgs += 1
        return total / (n_cfgs * 100.0)

    elif method == "sii_null":
        r_null = {s: min(services[s]["A"]) for s in others}
        vals   = [compute_kpis({**r_null, **dict(zip(S, rS))})[k]
                  for rS in cart_product(*A_S)]
        return (max(vals) - min(vals)) / 100.0

    elif method == "sobol":
        tv = _total_var[k]
        if tv == 0:
            return 0.0
        cond_means = []
        for rS in cart_product(*A_S):
            r_fix = dict(zip(S, rS))
            vals  = [compute_kpis({**r_fix, **dict(zip(others, ro))})[k]
                     for ro in (cart_product(*A_oth) if others else [()])]
            cond_means.append(np.mean(vals))
        return np.var(cond_means) / tv


# ── Individual sensitivity v({i}, k) ─────────────────────────────────────────
def compute_individual(method):
    out = np.zeros((n_tasks, n_kpis))
    for i, si in enumerate(task_ids):
        for l, k in enumerate(kpi_names):
            out[i, l] = v_func([si], k, method)
    return out


# ── Pairwise interaction I(i,j,k) ────────────────────────────────────────────
def compute_interaction(method, indiv):
    n_t = len(task_ids)
    I   = np.zeros((n_tasks, n_tasks, n_kpis))
    for i, si in enumerate(task_ids):
        for j, sj in enumerate(task_ids):
            if i >= j:
                continue
            others = [s for s in task_ids if s not in [si, sj]]
            for l, k in enumerate(kpi_names):
                if method in ("sii_marginalised", "sobol"):
                    I[i, j, l] = v_func([si, sj], k, method) - indiv[i, l] - indiv[j, l]

                elif method == "sii_null":
                    total = 0.0
                    for r in range(len(others) + 1):
                        for S_sub in icomb(others, r):
                            S_sub = list(S_sub)
                            w_shp = (factorial(len(S_sub))
                                     * factorial(n_t - len(S_sub) - 2)
                                     / factorial(n_t - 1))
                            d2    = (v_func(S_sub + [si, sj], k, method)
                                   - v_func(S_sub + [si],     k, method)
                                   - v_func(S_sub + [sj],     k, method)
                                   + v_func(S_sub,             k, method))
                            total += w_shp * d2
                    I[i, j, l] = total

            I[j, i, :] = I[i, j, :]
    return I


# ── F_pull ────────────────────────────────────────────────────────────────────
def fpull_from_interaction(I):
    raw  = np.einsum("ijk,k->ij", np.clip(I, 0, None), omega)
    mask = ~np.eye(n_tasks, dtype=bool)
    mx   = raw[mask].max()
    normed = raw / mx if mx > 0 else raw
    np.fill_diagonal(normed, 1.0)
    return normed


def fpull_cosine(indiv):
    C_vec  = indiv * omega[np.newaxis, :]
    norms  = np.linalg.norm(C_vec, axis=1, keepdims=True)
    C_norm = C_vec / np.where(norms > 0, norms, 1)
    fp     = C_norm @ C_norm.T
    np.fill_diagonal(fp, 1.0)
    return fp


# ── F_push (identical for all methods) ───────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


F_push = np.zeros((n_tasks, n_tasks))
for i, si in enumerate(task_ids):
    for j, sj in enumerate(task_ids):
        C_ij         = len(services[si]["A"]) * len(services[sj]["A"])
        F_push[i, j] = sigmoid(rho * (C_ij - C_max))
np.fill_diagonal(F_push, 0.0)


# ── Compute & save all methods ────────────────────────────────────────────────
SCENARIO_DIR = Path("results/analytical") / SCENARIO
BASE_OUT     = SCENARIO_DIR / "tig"
BASE_OUT.mkdir(parents=True, exist_ok=True)

shutil.copy(CONFIG_PATH, SCENARIO_DIR / "config.yaml")

scenario_meta = {
    "scenario":  SCENARIO,
    "config":    str(CONFIG_PATH.resolve()),
    "N": N, "M": M, "L": L, "delta": dlt,
    "C_max": C_max, "rho": rho,
    "tasks":       task_names,
    "kpis":        kpi_names,
    "kpi_weights": kpi_weights,
}
with open(BASE_OUT / "meta.json", "w") as fh:
    json.dump(scenario_meta, fh, indent=2)

for method in METHODS:
    print(f"\n[{method}] computing...", flush=True)

    v_method = "sii_marginalised" if method == "cosine" else method
    indiv    = compute_individual(v_method)
    print(f"  individual sensitivity done")

    if method == "cosine":
        F_pull = fpull_cosine(indiv)
        I      = np.zeros((n_tasks, n_tasks, n_kpis))
    else:
        I      = compute_interaction(method, indiv)
        F_pull = fpull_from_interaction(I)
    print(f"  F_pull done")

    assert np.allclose(F_pull, F_pull.T), "F_pull not symmetric"

    W = F_pull * (1.0 - F_push)
    np.fill_diagonal(W, 0.0)

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir = BASE_OUT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    df_W = pd.DataFrame(W,      index=task_names, columns=task_names)
    df_P = pd.DataFrame(F_pull, index=task_names, columns=task_names)
    df_X = pd.DataFrame(F_push, index=task_names, columns=task_names)
    df_v = pd.DataFrame(indiv,  index=task_names, columns=kpi_names)

    df_W.to_csv(out_dir / "W.csv")
    df_P.to_csv(out_dir / "F_pull.csv")
    df_X.to_csv(out_dir / "F_push.csv")
    df_v.to_csv(out_dir / "indiv.csv")

    # Same-node vs cross-node summary
    sn = np.mean([W[i, j] for i in range(n_tasks) for j in range(n_tasks)
                  if i != j and services[task_ids[i]]["node"] == services[task_ids[j]]["node"]])
    cn = np.mean([W[i, j] for i in range(n_tasks) for j in range(n_tasks)
                  if i != j and services[task_ids[i]]["node"] != services[task_ids[j]]["node"]])
    print(f"  saved → {out_dir}")
    print(f"  same-node W avg={sn:.4f}  cross-node W avg={cn:.4f}  gap={sn-cn:.4f}")

print(f"\nDone. Results in {SCENARIO_DIR}/ ({SCENARIO})")
