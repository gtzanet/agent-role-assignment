"""
Partition the Task Interaction Graph into agent groups.

Reads the W matrix produced by compute_analytical_tig.py and applies a
partitioning algorithm to assign tasks to agents.

TIG algorithms (require --method):
  greedy_modularity   maximises within-community W; n_agents inferred
  spectral            eigen-decomposition of W; requires --n-agents
  kernighan_lin       heuristic bisection; --n-agents must be 2

Topology baselines (--method ignored):
  all_in_one          single agent for all tasks
  all_separate        one agent per task
  per_node            one agent per compute node
  per_workflow        one agent per workflow

Outputs under results/analytical/<scenario>/partitions/<algo>[_<method>]/:
  assignment.json     {agent_id: [service_ids]}
  summary.json        metadata, complexity check, cut quality
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from typing import Optional
from networkx.algorithms.community import (
    greedy_modularity_communities,
    kernighan_lin_bisection,
)
from sklearn.cluster import SpectralClustering


TIG_ALGOS  = ["greedy_modularity", "spectral", "kernighan_lin"]
TOPO_ALGOS = ["all_in_one", "all_separate", "per_node", "per_workflow"]
ALL_ALGOS  = TIG_ALGOS + TOPO_ALGOS

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",    default="analytical_config.yaml",
                    help="Path to config YAML (default: analytical_config.yaml)")
parser.add_argument("--algorithm", required=True, choices=ALL_ALGOS)
parser.add_argument("--method",    default=None,
                    choices=["cosine", "sii_marginalised", "sii_null", "sobol"],
                    help="TIG delta method to load W from (required for TIG algorithms; "
                         "defaults to tig.delta_method in config)")
parser.add_argument("--n-agents",  type=int, default=None,
                    help="Number of agents (required for spectral; must be 2 for "
                         "kernighan_lin; defaults to partitioning.n_agents in config)")
args = parser.parse_args()

CONFIG_PATH = Path(args.config)

# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

SCENARIO = cfg.get("name", CONFIG_PATH.stem)

N = cfg["system"]["N"]
M = cfg["system"]["M"]

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
            "A":        svc["action_space"],
        }
        workflows[wid]["services"].append(sid)

node_services: dict = {n: [] for n in range(N)}
for sid, svc in services.items():
    node_services[svc["node"]].append(sid)

C_max = cfg["tig"]["C_max"]
task_ids   = sorted(services.keys())
task_names = [f"t{i}(s{i})" for i in task_ids]
n_tasks    = len(task_ids)

# Resolve --method and --n-agents from config defaults when not given
algorithm = args.algorithm

method = args.method
if method is None and algorithm in TIG_ALGOS:
    method = cfg.get("tig", {}).get("delta_method")
    if method is None:
        parser.error(f"--method is required for '{algorithm}' and no tig.delta_method in config")

n_agents = args.n_agents
if n_agents is None:
    n_agents = cfg.get("partitioning", {}).get("n_agents", 2)

if algorithm == "spectral" and n_agents is None:
    parser.error("--n-agents is required for spectral")
if algorithm == "kernighan_lin" and n_agents != 2:
    parser.error("kernighan_lin only supports n_agents=2")

# ── Load W matrix (TIG algorithms only) ──────────────────────────────────────
SCENARIO_DIR = Path("results/analytical") / SCENARIO
W            = None

if algorithm in TIG_ALGOS:
    w_path = SCENARIO_DIR / "tig" / method / "W.csv"
    if not w_path.exists():
        raise FileNotFoundError(
            f"W matrix not found at {w_path}\n"
            f"Run compute_analytical_tig.py first."
        )
    W = pd.read_csv(w_path, index_col=0).values
    print(f"Loaded W from: {w_path}  shape={W.shape}")


# ── Partition algorithms ──────────────────────────────────────────────────────
def run_partition(algo: str, W_mat: Optional[np.ndarray], n_ag: int) -> list[list[int]]:
    if algo == "greedy_modularity":
        G     = nx.from_numpy_array(W_mat)
        comms = list(greedy_modularity_communities(G, weight="weight"))
        return [sorted(c) for c in sorted(comms, key=lambda c: min(c))]

    if algo == "spectral":
        sc  = SpectralClustering(n_clusters=n_ag, affinity="precomputed",
                                 assign_labels="kmeans", random_state=0)
        lbl = sc.fit_predict(W_mat)
        comms = [[i for i, l in enumerate(lbl) if l == a] for a in range(n_ag)]
        return [sorted(c) for c in sorted(comms, key=lambda c: min(c))]

    if algo == "kernighan_lin":
        G    = nx.from_numpy_array(W_mat)
        a, b = kernighan_lin_bisection(G, weight="weight")
        return [sorted(a), sorted(b)]

    if algo == "all_in_one":
        return [list(range(n_tasks))]

    if algo == "all_separate":
        return [[i] for i in range(n_tasks)]

    if algo == "per_node":
        groups: dict = {}
        for idx, sid in enumerate(task_ids):
            groups.setdefault(services[sid]["node"], []).append(idx)
        return [sorted(v) for v in sorted(groups.values(), key=lambda g: min(g))]

    if algo == "per_workflow":
        groups = {}
        for idx, sid in enumerate(task_ids):
            groups.setdefault(services[sid]["workflow"], []).append(idx)
        return [sorted(v) for v in sorted(groups.values(), key=lambda g: min(g))]

    raise ValueError(f"Unknown algorithm: {algo}")


communities   = run_partition(algorithm, W, n_agents)
task_to_agent = {t: a for a, grp in enumerate(communities) for t in grp}

# ── Report ────────────────────────────────────────────────────────────────────
label = f"{algorithm}" + (f"_{method}" if algorithm in TIG_ALGOS else "")
print(f"\nScenario  : {SCENARIO}")
print(f"Algorithm : {algorithm}" + (f"  method={method}" if algorithm in TIG_ALGOS else ""))
print(f"Groups    : {len(communities)}\n")

agent_records = []
for agent_id, group in enumerate(communities):
    sids        = [task_ids[t] for t in group]
    action_sizes = [len(services[s]["A"]) for s in sids]
    joint_space  = 1
    for sz in action_sizes:
        joint_space *= sz
    ok = joint_space <= C_max
    flag = "✓" if ok else f"✗ EXCEEDS C_max={C_max}"
    print(f"  Agent {agent_id}: {[task_names[t] for t in group]}")
    print(f"    nodes={[services[task_ids[t]]['node'] for t in group]}"
          f"  workflows={[services[task_ids[t]]['workflow'] for t in group]}")
    print(f"    |A|^k = {'×'.join(str(s) for s in action_sizes)} = {joint_space}  {flag}\n")
    agent_records.append({
        "agent_id":    agent_id,
        "service_ids": sids,
        "nodes":       [services[s]["node"]    for s in sids],
        "workflows":   [services[s]["workflow"] for s in sids],
        "action_sizes": action_sizes,
        "joint_space": joint_space,
        "within_C_max": ok,
    })

# ── Cut quality (TIG algorithms only) ────────────────────────────────────────
cut_stats = {}
if W is not None:
    within = [W[i, j] for i in range(n_tasks) for j in range(i + 1, n_tasks)
              if task_to_agent[i] == task_to_agent[j]]
    cut    = [W[i, j] for i in range(n_tasks) for j in range(i + 1, n_tasks)
              if task_to_agent[i] != task_to_agent[j]]
    if within:
        print(f"Within-group W : mean={np.mean(within):.4f}  min={np.min(within):.4f}")
    if cut:
        print(f"Cut-edge     W : mean={np.mean(cut):.4f}  max={np.max(cut):.4f}")
    cut_stats = {
        "within_mean": round(float(np.mean(within)), 4) if within else None,
        "within_min":  round(float(np.min(within)),  4) if within else None,
        "cut_mean":    round(float(np.mean(cut)),     4) if cut    else None,
        "cut_max":     round(float(np.max(cut)),      4) if cut    else None,
    }

# ── Save outputs ──────────────────────────────────────────────────────────────
OUT_DIR = SCENARIO_DIR / "partitions" / label
OUT_DIR.mkdir(parents=True, exist_ok=True)

assignment = {str(rec["agent_id"]): rec["service_ids"] for rec in agent_records}
with open(OUT_DIR / "assignment.json", "w") as fh:
    json.dump(assignment, fh, indent=2)

summary = {
    "scenario":   SCENARIO,
    "algorithm":  algorithm,
    "method":     method,
    "n_agents_requested": n_agents,
    "n_agents_found":     len(communities),
    "C_max":      C_max,
    "agents":     agent_records,
    "cut_stats":  cut_stats,
    "all_within_C_max": all(rec["within_C_max"] for rec in agent_records),
}
with open(OUT_DIR / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\nOutputs → {OUT_DIR}/")
for f_out in sorted(OUT_DIR.iterdir()):
    print(f"  {f_out.name}")
