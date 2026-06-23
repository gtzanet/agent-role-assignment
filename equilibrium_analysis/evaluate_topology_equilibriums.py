"""
Evaluate Nash equilibria for all topologies in a dataset.

All system parameters are loaded from a YAML config file.

Usage:
    python3 equilibrium_analysis/evaluate_topology_equilibriums.py --topologies equilibrium_analysis/topologies/topologies_M1-5_N1-5_cap100_seed42.json
    python3 equilibrium_analysis/evaluate_topology_equilibriums.py --topologies <path> --config <path>

Output: equilibrium_analysis/results/<YYYYMMDD_HHMMSS>/
    equilibriums.json  — per-topology, per-algo NE metrics
    config.yaml        — copy of the config used
    topologies.json    — copy of the topologies dataset used
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ── Project-root imports (gps_model, partitioning, evaluate_equilibriums) ────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gps_model import GPSModel
from partitioning import partition_task_ids
from evaluate_equilibriums import TaskInteractionGraph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _action_space(M: int, as_cfg: dict) -> list[int]:
    if M <= 2:
        return list(as_cfg["m_le_2"])
    elif M == 3:
        return list(as_cfg["m_eq_3"])
    else:
        return list(as_cfg["m_ge_4"])


def _build_cfg(M: int, N: int, mapping: dict, params: dict) -> dict:
    action_space = _action_space(M, params["action_space"])
    sid = 0
    workflows = []
    for w in range(M):
        services_cfg = []
        for s in range(params["services_per_wf"]):
            node_id = int(mapping[f"w{w}_s{s}"][1:])
            services_cfg.append({
                "id": sid,
                "node": node_id,
                "replicas_baseline": params["replicas_base"],
                "action_space": action_space,
            })
            sid += 1
        workflows.append({"id": w, "lambda": params["lambda_per_wf"], "services": services_cfg})

    return {
        "name": f"topo_M{M}_N{N}",
        "system": {"N": N, "M": M, "L": params["task_load"]},
        "nodes": [{"id": n, "f": params["node_freq"]} for n in range(N)],
        "delta": params["delta"],
        "workflows": workflows,
        "tig": params["tig_cfg"],
        "partitioning": {"algorithm": "per_node", "n_agents": N},
    }


# ── Equilibrium enumeration ───────────────────────────────────────────────────

def _enumerate_equilibria(model: GPSModel, assignment: dict[int, list[int]]):
    """Return (profile_list, equilibria). O(n) fast path for single-agent games."""
    agent_ids = sorted(assignment.keys())

    agent_actions: dict[int, list[dict]] = {}
    for aid in agent_ids:
        sids = assignment[aid]
        agent_actions[aid] = [
            dict(zip(sids, combo))
            for combo in itertools.product(*[model.services[s]["A"] for s in sids])
        ]

    matrix: dict[tuple, dict] = {}
    profile_list: list[dict] = []

    for idx_tuple in itertools.product(*(range(len(agent_actions[a])) for a in agent_ids)):
        r = dict(model.baseline_assignment)
        for aid_i, aid in enumerate(agent_ids):
            r.update(agent_actions[aid][idx_tuple[aid_i]])

        stats = model.compute_service_stats(r)

        agent_utils: dict[int, dict] = {}
        for aid in agent_ids:
            sids = assignment[aid]
            u1 = sum(1 for s in sids if stats["rho_s"][s] < 1.0) / len(sids) if sids else 0.0
            u2 = sum(max(0.0, stats["lam_eff"][s] - stats["C"][s]) for s in sids)
            agent_utils[aid] = {"U1": u1, "U2": u2}

        wids = list(model.workflows.keys())
        g_u1 = sum(
            1 for wid in wids
            if model.workflows[wid]["lambda"] < min(
                stats["C"][s] for s in model.workflows[wid]["services"]
            )
        ) / len(wids)
        g_u2 = sum(max(0.0, stats["lam_eff"][s] - stats["C"][s]) for s in model.task_ids)
        throughput = sum(min(stats["lam_eff"][s], stats["C"][s]) for s in model.task_ids)

        profile = {
            "indices":     idx_tuple,
            "replicas":    dict(r),
            "agent_utils": agent_utils,
            "global":      {"U1": g_u1, "U2": g_u2, "throughput": throughput},
        }
        matrix[idx_tuple] = profile
        profile_list.append(profile)

    # Single-agent fast path: NE = lexicographic global optimum
    if len(agent_ids) == 1:
        aid    = agent_ids[0]
        max_u1 = max(p["agent_utils"][aid]["U1"] for p in profile_list)
        cands  = [p for p in profile_list if p["agent_utils"][aid]["U1"] == max_u1]
        min_u2 = min(p["agent_utils"][aid]["U2"] for p in cands)
        return profile_list, [p for p in cands if p["agent_utils"][aid]["U2"] == min_u2]

    equilibria: list[dict] = []
    for idx_tuple, profile in matrix.items():
        is_ne = True
        for aid_i, aid in enumerate(agent_ids):
            cur = profile["agent_utils"][aid]
            for dev_i in range(len(agent_actions[aid])):
                if dev_i == idx_tuple[aid_i]:
                    continue
                dev_idx = list(idx_tuple)
                dev_idx[aid_i] = dev_i
                dev = matrix[tuple(dev_idx)]["agent_utils"][aid]
                if dev["U1"] > cur["U1"] or (dev["U1"] == cur["U1"] and dev["U2"] < cur["U2"]):
                    is_ne = False
                    break
            if not is_ne:
                break
        if is_ne:
            equilibria.append(profile)

    return profile_list, equilibria


def _poa(profile_list: list[dict], equilibria: list[dict]) -> float | None:
    if not equilibria or not profile_list:
        return None
    w_opt   = max(p["global"]["throughput"] for p in profile_list)
    w_worst = min(e["global"]["throughput"] for e in equilibria)
    if w_worst <= 0:
        return float("inf")
    return w_opt / w_worst


def _metrics_for_assign(model: GPSModel, assign: dict) -> dict:
    profile_list, equilibria = _enumerate_equilibria(model, assign)
    poa = _poa(profile_list, equilibria)

    metrics: dict = {
        "assignment":   {aid: sids for aid, sids in assign.items()},
        "n_agents":     len(assign),
        "n_profiles":   len(profile_list),
        "n_equilibria": len(equilibria),
        "PoA":          round(poa, 6) if poa is not None else None,
    }
    if equilibria:
        worst = min(equilibria, key=lambda e: (e["global"]["U1"], -e["global"]["U2"]))
        metrics["worst_U1"] = round(worst["global"]["U1"], 6)
        metrics["worst_U2"] = round(worst["global"]["U2"], 6)
        metrics["avg_U1"]   = round(sum(e["global"]["U1"] for e in equilibria) / len(equilibria), 6)
        metrics["avg_U2"]   = round(sum(e["global"]["U2"] for e in equilibria) / len(equilibria), 6)
    else:
        metrics.update(worst_U1=None, worst_U2=None, avg_U1=None, avg_U2=None)
    return metrics


def _get_assign(model: GPSModel, algo: str, gps_cfg: dict, N: int,
                tig_label: str, tig_algo: str) -> dict:
    if algo == tig_label:
        W = TaskInteractionGraph.from_dict(gps_cfg).compute_W()
        return partition_task_ids(model, algorithm=tig_algo, n_agents=N, w=W)
    return partition_task_ids(model, algorithm=algo, n_agents=N)


def evaluate_topology(M: int, N: int, mapping: dict, idx: int, params: dict) -> dict:
    tig_label = f"tig_{params['tig_algo']}"
    gps_cfg   = _build_cfg(M, N, mapping, params)
    model     = GPSModel(gps_cfg)

    result: dict = {"topology_idx": idx, "mapping": mapping}
    algos = [
        a for a in params["assignment_algos"]
        if not (a == tig_label and M > params["tig_max_m"])
    ]
    for algo in algos:
        assign       = _get_assign(model, algo, gps_cfg, N, tig_label, params["tig_algo"])
        result[algo] = _metrics_for_assign(model, assign)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate topology equilibria. All parameters via config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPT_DIR / "config.yaml",
        help="Path to YAML config (default: equilibrium_analysis/config.yaml)",
    )
    parser.add_argument(
        "--topologies",
        type=Path,
        required=True,
        help="Path to topologies JSON file (e.g. equilibrium_analysis/topologies/...json)",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    params = _load_config(config_path)

    topo_path = args.topologies.resolve()

    with open(topo_path) as f:
        dataset = json.load(f)

    # ── Create experiment directory ───────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir   = _SCRIPT_DIR / "results" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, exp_dir / "config.yaml")
    shutil.copy(topo_path,   exp_dir / "topologies.json")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results: dict = {}

    for key, entry in dataset.items():
        M, N = entry["M"], entry["N"]
        entry_params = {**params, "services_per_wf": entry.get("services_per_wf", params["services_per_wf"])}
        results: list[dict] = []

        tag = (f" [sampled {entry['num_mappings']}/{entry['total_topologies']:,}]"
               if entry.get("sampled") else "")
        print(f"\n{'='*60}")
        print(f"{key}  M={M}  N={N}  "
              f"A={_action_space(M, params['action_space'])}  "
              f"({entry['num_mappings']} topologies{tag})")

        for idx, mapping in enumerate(entry["mappings"]):
            result = evaluate_topology(M, N, mapping, idx, entry_params)
            results.append(result)
            algos   = [a for a in params["assignment_algos"] if a in result]
            summary = "  ".join(
                f"{a.split('_')[0]}:NE={result[a]['n_equilibria']},PoA={result[a]['PoA']}"
                for a in algos
            )
            print(f"  [{idx:>3}] {summary}")

        all_results[key] = results

    out_path = exp_dir / "equilibriums.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nExperiment saved → {exp_dir}/")
    print(f"  equilibriums.json  ({out_path.stat().st_size:,} bytes)")
    print(f"  config.yaml")
    print(f"  topologies.json    ({topo_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
