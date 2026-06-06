"""
Coordinate-descent policy evaluation on the analytical GPS model.

Each agent iterates best-response simultaneously: every round all agents
read the current state, independently search their own action space, and
apply their decisions together.

Agent objective (per round):
  - Feasible combos (all ρ_wf < 1): minimise mean(ρ_s) over own services,
    where ρ_s = λ_eff_s / C_s (per-service utilisation / traffic intensity).
  - No feasible combo: pick the least-infeasible one — minimum max(ρ_wf),
    where ρ_wf = λ_w / C_{last_service_w}.

Convergence: no agent changes in a full round, or max_iter reached.

Usage:
    python evaluate_analytical_policy.py \\
        --partition spectral_sii_marginalised \\
        [--config analytical_config.yaml] \\
        [--max-iter 100]

Outputs under results/analytical/<scenario>/policy/<partition_label>/:
    trajectory.csv       per-round state and KPIs
    final_assignment.json
    final_metrics.csv    per-service and per-workflow stats at convergence
    summary.json
"""

import argparse
import json
from copy import deepcopy
from itertools import product as cart_product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from compute_analytical_metrics import GPSModel


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",    default="analytical_config.yaml")
parser.add_argument("--partition", required=True,
                    help="Partition label (subdir under partitions/), "
                         "e.g. spectral_sii_marginalised or per_workflow")
parser.add_argument("--max-iter",  type=int,   default=100)
parser.add_argument("--noise",     type=float, default=0.0,
                    help="Lambda noise amplitude ε ∈ [0, 0.5]. Each round each "
                         "workflow observes lambda * (1 + U(-ε, ε)), clamped to ≥ 0. "
                         "Agents decide on the noisy observation; KPIs are evaluated "
                         "with true lambdas. (default: 0.0 = no noise)")
parser.add_argument("--seed",      type=int,   default=None,
                    help="Random seed for noise draws (default: None)")
args = parser.parse_args()

if not 0.0 <= args.noise <= 0.5:
    parser.error("--noise must be in [0, 0.5]")

CONFIG_PATH = Path(args.config)
with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

SCENARIO    = cfg.get("name", CONFIG_PATH.stem)
model       = GPSModel(cfg)
MAX_ITER    = args.max_iter
NOISE_AMP   = args.noise

if args.seed is not None:
    np.random.seed(args.seed)

# ── Load partition assignment ─────────────────────────────────────────────────
SCENARIO_DIR = Path("results/analytical") / SCENARIO
PART_DIR     = SCENARIO_DIR / "partitions" / args.partition

if not PART_DIR.exists():
    raise FileNotFoundError(
        f"Partition not found: {PART_DIR}\n"
        f"Run partition_tig.py first."
    )

with open(PART_DIR / "assignment.json") as fh:
    raw = json.load(fh)

# {agent_id (int): [service_ids (int)]}
assignment = {int(k): [int(s) for s in v] for k, v in raw.items()}
n_agents   = len(assignment)

# Workflows "owned" by each agent: those whose leaf service is in the agent's assignment.
# Used by the infeasible fallback so agents only minimise ρ over what they control.
agent_owned_wfs: dict = {
    aid: [wid for wid, wf in model.workflows.items()
          if wf["services"][-1] in sids]
    for aid, sids in assignment.items()
}

print(f"Scenario  : {SCENARIO}")
print(f"Partition : {args.partition}  ({n_agents} agents)")
for aid, sids in sorted(assignment.items()):
    print(f"  Agent {aid}: services {sids}  "
          f"nodes={[model.services[s]['node'] for s in sids]}  "
          f"owns wfs={agent_owned_wfs[aid]}")


# ── Agent helpers ─────────────────────────────────────────────────────────────
def is_ergodic(rho_wf: dict) -> bool:
    return all(v < 1.0 for v in rho_wf.values())


def best_combo(agent_id: int, r_snapshot: dict,
               lambda_obs: Optional[dict] = None) -> tuple[tuple, bool]:
    """
    Returns (best_replica_tuple, feasible).
    Tuple is in the order of assignment[agent_id].
    Uses r_snapshot for all services not owned by this agent.
    lambda_obs: noisy per-workflow lambda observations for this round;
                None means use true lambdas.
    """
    sids = assignment[agent_id]
    A    = [model.services[s]["A"] for s in sids]

    feasible:   list = []   # (mean_rho_s, combo)
    infeasible: list = []   # (max_rho_wf, combo)

    for combo in cart_product(*A):
        r     = {**r_snapshot, **dict(zip(sids, combo))}
        stats = model.compute_service_stats(r, lambda_obs=lambda_obs)

        if is_ergodic(stats["rho_wf"]):
            obj = float(np.mean([stats["rho_s"][s] for s in sids]))
            feasible.append((obj, combo))
        else:
            owned = agent_owned_wfs[agent_id]
            worst = (max(stats["rho_wf"][w] for w in owned)
                     if owned else max(stats["rho_wf"].values()))
            infeasible.append((worst, combo))

    if feasible:
        return min(feasible, key=lambda x: x[0])[1], True
    return min(infeasible, key=lambda x: x[0])[1], False


# ── Initialise ────────────────────────────────────────────────────────────────
r = {sid: 1 for sid in model.task_ids}

stats0 = model.compute_service_stats(r)
kpis0  = model.compute_kpis(r)
def print_state(label: str, kpis: dict, stats: dict, r: dict):
    J = model.score(kpis)
    print(f"\n{label}  (ergodic={is_ergodic(stats['rho_wf'])}  J={J:.3f})")
    print(f"  r = { {f's{s}': r[s] for s in model.task_ids} }")
    for wid in range(model.M):
        drop = kpis[f"D_wf{wid}/λ"]
        rho  = stats["rho_wf"][wid]
        print(f"  wf{wid}:  drop={drop:5.1f}%  ρ_wf={rho:.3f}"
              + ("  OVERLOADED" if rho >= 1 else ""))
    for n in range(model.N):
        print(f"  node{n}: u={kpis[f'u_node{n}']:5.1f}%")

print_state("Initial state  (r=1 for all)", kpis0, stats0, r)
J_initial = model.score(kpis0)


# ── Coordinate-descent loop ───────────────────────────────────────────────────
trajectory = []
converged  = False

for iteration in range(1, MAX_ITER + 1):
    r_snap   = deepcopy(r)
    r_new    = deepcopy(r_snap)
    changed  = {}    # {agent_id: new_values_dict}
    feasible = {}    # {agent_id: bool}

    # Draw per-workflow noisy lambda observations for this round
    if NOISE_AMP > 0:
        lambda_obs = {
            wid: max(0.0, wf["lambda"] * (1.0 + np.random.uniform(-NOISE_AMP, NOISE_AMP)))
            for wid, wf in model.workflows.items()
        }
    else:
        lambda_obs = None

    for aid in sorted(assignment.keys()):
        combo, is_feas = best_combo(aid, r_snap, lambda_obs=lambda_obs)
        new_vals       = dict(zip(assignment[aid], combo))
        feasible[aid]  = is_feas

        if any(new_vals[s] != r_snap[s] for s in assignment[aid]):
            changed[aid] = new_vals
            r_new.update(new_vals)

    r = r_new

    stats = model.compute_service_stats(r)
    kpis  = model.compute_kpis(r)
    J     = model.score(kpis)

    row = {
        "iteration":      iteration,
        "agents_changed": str(sorted(changed.keys())),
        "n_changed":      len(changed),
        "ergodic":        is_ergodic(stats["rho_wf"]),
        "J":              round(J, 4),
    }
    for wid in range(model.M):
        row[f"rho_wf{wid}"] = round(stats["rho_wf"][wid], 4)
        if lambda_obs is not None:
            row[f"lambda_obs_wf{wid}"] = round(lambda_obs[wid], 4)
    for n in range(model.N):
        row[f"u_node{n}"] = round(kpis[f"u_node{n}"], 2)
    for sid in model.task_ids:
        row[f"r_s{sid}"] = r[sid]
    trajectory.append(row)

    drop_str = "  ".join(f"wf{w}:{kpis[f'D_wf{w}/λ']:4.1f}%" for w in range(model.M))
    util_str = "  ".join(f"n{n}:{kpis[f'u_node{n}']:4.1f}%" for n in range(model.N))
    lobs_str = ("  ".join(
                    f"wf{w}:{lambda_obs[w]:.1f}(true:{model.workflows[w]['lambda']:.1f})"
                    for w in range(model.M))
                if lambda_obs is not None else None)
    print(f"  iter {iteration:3d}:  {len(changed)} changed"
          + (f"  agents={sorted(changed.keys())}" if changed else "")
          + f"  ergodic={row['ergodic']}  J={J:.3f}"
          + (f"\n         λ_obs: {lobs_str}" if lobs_str else "")
          + f"\n         drop: {drop_str}"
          + f"\n         util: {util_str}"
          + f"\n         r={ {f's{s}': r[s] for s in model.task_ids} }")

    if not changed:
        converged = True
        print(f"\nConverged at iteration {iteration}.")
        break

if not converged:
    print(f"\nMax iterations ({MAX_ITER}) reached without convergence.")

print(f"Rounds: {len(trajectory)}")


# ── Final report ──────────────────────────────────────────────────────────────
stats_f = model.compute_service_stats(r)
kpis_f  = model.compute_kpis(r)
J_f     = model.score(kpis_f)

print(f"\n{'─'*60}")
print_state("FINAL", kpis_f, stats_f, r)
for sid in model.task_ids:
    print(f"  s{sid}: ρ_s={stats_f['rho_s'][sid]:.4f}  "
          f"C={stats_f['C'][sid]:.2f}  λ_eff={stats_f['lam_eff'][sid]:.2f}")


# ── Save outputs ──────────────────────────────────────────────────────────────
OUT_DIR = SCENARIO_DIR / "policy" / args.partition
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(trajectory).to_csv(OUT_DIR / "trajectory.csv", index=False)

with open(OUT_DIR / "final_assignment.json", "w") as fh:
    json.dump({f"s{s}": int(r[s]) for s in model.task_ids}, fh, indent=2)

metrics_rows = []
for sid in model.task_ids:
    metrics_rows.append({
        "type":     "service",
        "id":       f"s{sid}",
        "node":     model.services[sid]["node"],
        "workflow": model.services[sid]["workflow"],
        "replicas": r[sid],
        "C":        round(stats_f["C"][sid], 4),
        "lam_eff":  round(stats_f["lam_eff"][sid], 4),
        "rho":      round(stats_f["rho_s"][sid], 4),
    })
for wid, wf in model.workflows.items():
    metrics_rows.append({
        "type":     "workflow",
        "id":       f"wf{wid}",
        "node":     "",
        "workflow": wid,
        "replicas": "",
        "C":        round(stats_f["C"][wf["services"][-1]], 4),
        "lam_eff":  round(wf["lambda"], 4),
        "rho":      round(stats_f["rho_wf"][wid], 4),
    })
pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "final_metrics.csv", index=False)

summary = {
    "scenario":          SCENARIO,
    "partition":         args.partition,
    "max_iter":          MAX_ITER,
    "noise_amp":         NOISE_AMP,
    "seed":              args.seed,
    "converged":         converged,
    "iterations":        len(trajectory),
    "ergodic":           is_ergodic(stats_f["rho_wf"]),
    "J_initial":         round(J_initial, 4),
    "J_final":           round(J_f, 4),
    "final_assignment":  {f"s{s}": int(r[s]) for s in model.task_ids},
    "rho_wf":            {f"wf{w}": round(v, 4) for w, v in stats_f["rho_wf"].items()},
    "rho_s":             {f"s{s}": round(v, 4) for s, v in stats_f["rho_s"].items()},
}
with open(OUT_DIR / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\nOutputs → {OUT_DIR}/")
for f_out in sorted(OUT_DIR.iterdir()):
    print(f"  {f_out.name}")
