"""
Evaluate Nash equilibria for all topologies in a dataset.

All system parameters are loaded from a YAML config file.

lambda_per_wf may be a single number, a list of numbers, or
{"capacity_fractions": [...]} to derive lambda dynamically per topology
(M, N) as a target fraction of total system capacity (see _lambda_values).
Every topology is evaluated once per resulting lambda value (cross product),
and each result entry records which lambda it was evaluated under.

Usage:
    python3 equilibrium_analysis/evaluate_topology_equilibriums.py --topologies equilibrium_analysis/topologies/topologies_M1-5_N1-5_cap100_seed42.json
    python3 equilibrium_analysis/evaluate_topology_equilibriums.py --topologies <path> --config <path>

Output: equilibrium_analysis/results/<config-name>/
    equilibriums.json  — per-topology, per-lambda, per-algo NE metrics
    config.yaml        — copy of the config used
    topologies.json    — copy of the topologies dataset used

<config-name> is derived from the run's configuration (topology dataset,
lambda, equilibrium metric, latency threshold, admission-control mode) --
see _exp_name() -- so re-running the same configuration overwrites its own
directory instead of piling up timestamped ones, mirroring how
generate_topology_dataset.py names its topology files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
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
        "system": {
            "N": N, "M": M, "L": params["task_load"],
            "drop_on_overload": params.get("drop_on_overload", True),
            "latency_threshold_pct": params.get("latency_threshold_pct", 200),
            "u_metric": params.get("u_metric", "utilization"),
        },
        "nodes": [{"id": n, "f": params["node_freq"]} for n in range(N)],
        "delta": params["delta"],
        "workflows": workflows,
        "tig": params["tig_cfg"],
        "partitioning": {"algorithm": "per_node", "n_agents": N},
    }


# ── Equilibrium enumeration ───────────────────────────────────────────────────

def _enumerate_equilibria(model: GPSModel, assignment: dict[int, list[int]],
                           equilibrium_metric: str = "drop"):
    """Return (profile_list, equilibria). O(n) fast path for single-agent games.

    equilibrium_metric selects how the reported global utility U (used for
    worst/avg-NE utility and PoA) is computed -- it does not affect which
    profiles are Nash equilibria, since that's driven purely by each agent's
    own utility (mean effective capacity C_s over its services -- always
    replica-dependent, regardless of model.drop_on_overload):
      "drop"            -- avg over workflows of min(1, C_bottleneck/lambda_w),
                            i.e. avg(1 - D_w/lambda_w). Mode-independent
                            (structural, same regardless of drop_on_overload).
      "latency_success" -- avg over workflows of the binary latency-SLA
                            success indicator sigma_w (see
                            GPSModel.workflow_latency_success).
    """
    if equilibrium_metric not in ("drop", "latency_success"):
        raise ValueError(
            f"equilibrium_metric must be 'drop' or 'latency_success', got {equilibrium_metric!r}"
        )
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
            u = (sum(stats["C"][s] for s in sids) / len(sids)) if sids else 0.0
            agent_utils[aid] = {"U": u}

        wids = list(model.workflows.keys())
        if equilibrium_metric == "latency_success":
            sigma_w = model.workflow_latency_success(r)
            g_u = sum(sigma_w[wid] for wid in wids) / len(wids)
        else:
            g_u = sum(
                min(1.0, min(stats["C"][s] for s in model.workflows[wid]["services"])
                    / model.workflows[wid]["lambda"])
                if model.workflows[wid]["lambda"] > 0 else 1.0
                for wid in wids
            ) / len(wids)

        profile = {
            "indices":     idx_tuple,
            "replicas":    dict(r),
            "agent_utils": agent_utils,
            "global":      {"U": g_u},
        }
        matrix[idx_tuple] = profile
        profile_list.append(profile)

    # Single-agent fast path: NE = profile(s) with maximum agent U
    if len(agent_ids) == 1:
        aid   = agent_ids[0]
        max_u = max(p["agent_utils"][aid]["U"] for p in profile_list)
        return profile_list, [p for p in profile_list if p["agent_utils"][aid]["U"] == max_u]

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
                if dev["U"] > cur["U"]:
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
    w_opt   = max(p["global"]["U"] for p in profile_list)
    w_worst = min(e["global"]["U"] for e in equilibria)
    if w_worst <= 0:
        return float("inf")
    return w_opt / w_worst


def _metrics_for_assign(model: GPSModel, assign: dict, equilibrium_metric: str = "drop") -> dict:
    profile_list, equilibria = _enumerate_equilibria(model, assign, equilibrium_metric)
    poa = _poa(profile_list, equilibria)

    metrics: dict = {
        "assignment":   {aid: sids for aid, sids in assign.items()},
        "n_agents":     len(assign),
        "n_profiles":   len(profile_list),
        "n_equilibria": len(equilibria),
        "PoA":          round(poa, 6) if poa is not None else None,
    }
    if equilibria:
        worst = min(equilibria, key=lambda e: e["global"]["U"])
        metrics["worst_U"] = round(worst["global"]["U"], 6)
        metrics["avg_U"]   = round(sum(e["global"]["U"] for e in equilibria) / len(equilibria), 6)
    else:
        metrics.update(worst_U=None, avg_U=None)
    return metrics


def _get_assign(model: GPSModel, algo: str, gps_cfg: dict, N: int,
                tig_label: str, tig_algo: str) -> dict:
    if algo == tig_label:
        W = TaskInteractionGraph.from_dict(gps_cfg).compute_W()
        return partition_task_ids(model, algorithm=tig_algo, n_agents=N, w=W)
    return partition_task_ids(model, algorithm=algo, n_agents=N)


def _evaluate_one(task: tuple) -> dict:
    """Picklable worker entry point for one topology (ProcessPoolExecutor)."""
    M, N, mapping, idx, params = task
    return evaluate_topology(M, N, mapping, idx, params)


def _resolve_workers(cli_workers: int | None, params: dict) -> int:
    """Bounded worker count. Caps peak RAM at ~workers × one topology's matrix."""
    if cli_workers is not None and cli_workers > 0:
        return cli_workers
    cfg = params.get("max_workers")
    if cfg and cfg > 0:
        return int(cfg)
    return max(1, (os.cpu_count() or 2) - 1)


def _lambda_desc(lam) -> str:
    if isinstance(lam, dict):
        return "capfrac" + "-".join(f"{f:g}" for f in lam["capacity_fractions"])
    values = lam if isinstance(lam, list) else [lam]
    return "lam" + "-".join(f"{v:g}" for v in values)


def _exp_name(params: dict, topo_path: Path) -> str:
    """Configuration-derived name for the results directory, mirroring how
    generate_topology_dataset.py names topology files. Two runs with the same
    topology dataset, lambda, equilibrium metric, latency threshold, and
    admission-control mode share a directory (the later run overwrites it) --
    those are the parameters observed to actually change results across runs.
    """
    topo_stem = topo_path.stem
    if topo_stem.startswith("topologies_"):
        topo_stem = topo_stem[len("topologies_"):]
    return "_".join([
        topo_stem,
        _lambda_desc(params["lambda_per_wf"]),
        f"eq-{params.get('equilibrium_metric', 'drop')}",
        f"lat{params.get('latency_threshold_pct', 0)}",
        f"adm{int(bool(params.get('drop_on_overload', False)))}",
    ])


def _lambda_values(params: dict, M: int, N: int) -> list[tuple[float, float | None]]:
    """lambda_per_wf may be:
      - a single number
      - a list of numbers
      - {"capacity_fractions": [...]} to derive lambda values (e.g. low/medium/
        high) from a target fraction of total system capacity, per topology
        (M, N differ per dataset entry):
            total workload = lambda * M * services_per_wf * task_load
            total capacity = N * node_freq
        Solving workload = fraction * capacity for lambda:
            lambda = fraction * (N * node_freq) / (M * services_per_wf * task_load)

    Returns a list of (lambda_value, capacity_fraction) pairs. capacity_fraction
    is None unless lambda_per_wf uses the capacity_fractions form -- it's the
    only thing that stays comparable across topologies with different M/N,
    since the resolved lambda value itself differs per topology even for the
    "same" scenario.
    """
    lam = params["lambda_per_wf"]
    if isinstance(lam, dict):
        fractions = lam["capacity_fractions"]
        denom = M * params["services_per_wf"] * params["task_load"]
        return [(frac * (N * params["node_freq"]) / denom, frac) for frac in fractions]
    values = list(lam) if isinstance(lam, list) else [lam]
    return [(v, None) for v in values]


def _print_progress(result: dict, idx: int, params: dict) -> None:
    algos   = [a for a in params["assignment_algos"] if a in result]
    summary = "  ".join(
        f"{a.split('_')[0]}:NE={result[a]['n_equilibria']},PoA={result[a]['PoA']}"
        for a in algos
    )
    frac    = result.get("lambda_capacity_fraction")
    lam_str = f" λ={result['lambda']:g}" + (f"({frac:g}x)" if frac is not None else "")
    print(f"  [{idx:>3}]{lam_str} {summary}")


def evaluate_topology(M: int, N: int, mapping: dict, idx: int, params: dict) -> dict:
    tig_label = f"tig_{params['tig_algo']}"
    gps_cfg   = _build_cfg(M, N, mapping, params)
    model     = GPSModel(gps_cfg)

    result: dict = {
        "topology_idx": idx,
        "lambda": params["lambda_per_wf"],
        "lambda_capacity_fraction": params.get("lambda_capacity_fraction"),
        "mapping": mapping,
    }
    algos = [
        a for a in params["assignment_algos"]
        if not (a == tig_label and M > params["tig_max_m"])
    ]
    equilibrium_metric = params.get("equilibrium_metric", "drop")
    for algo in algos:
        assign       = _get_assign(model, algo, gps_cfg, N, tig_label, params["tig_algo"])
        result[algo] = _metrics_for_assign(model, assign, equilibrium_metric)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes. Default: config max_workers, else cpu_count-1. "
             "Use 1 to run serially. Lower this if you hit memory limits.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    params = _load_config(config_path)

    topo_path = args.topologies.resolve()

    with open(topo_path) as f:
        dataset = json.load(f)

    # ── Create experiment directory ───────────────────────────────────────────
    exp_dir = _SCRIPT_DIR / "results" / _exp_name(params, topo_path)
    exp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, exp_dir / "config.yaml")
    shutil.copy(topo_path,   exp_dir / "topologies.json")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    max_workers = _resolve_workers(args.workers, params)

    # Recycle workers periodically so the large transient profile matrices are
    # returned to the OS instead of accumulating (Python 3.11+ only).
    pool_kwargs: dict = {"max_workers": max_workers}
    if sys.version_info >= (3, 11):
        pool_kwargs["max_tasks_per_child"] = int(params.get("max_tasks_per_child", 50))

    print(f"Parallelism: {max_workers} worker process(es)"
          + (" (serial)" if max_workers == 1 else ""))

    all_results: dict = {}
    out_path = exp_dir / "equilibriums.json"

    for key, entry in dataset.items():
        M, N = entry["M"], entry["N"]
        services_per_wf = entry.get("services_per_wf")
        if services_per_wf is None:
            services_per_wf = params["services_per_wf"]
        entry_params  = {**params, "services_per_wf": services_per_wf}
        lambda_values = _lambda_values(entry_params, M, N)

        tag = (f" [sampled {entry['num_mappings']}/{entry['total_topologies']:,}]"
               if entry.get("sampled") else "")
        lam_summary = ", ".join(
            f"{lam:g}" + (f"({frac:g}x)" if frac is not None else "")
            for lam, frac in lambda_values
        )
        print(f"\n{'='*60}")
        print(f"{key}  M={M}  N={N}  "
              f"A={_action_space(M, params['action_space'])}  "
              f"({entry['num_mappings']} topologies{tag})  "
              f"λ=[{lam_summary}]")

        tasks = [
            (M, N, mapping, idx, {**entry_params, "lambda_per_wf": lam, "lambda_capacity_fraction": frac})
            for idx, mapping in enumerate(entry["mappings"])
            for lam, frac in lambda_values
        ]

        if max_workers == 1:
            results = []
            for task in tasks:
                result = _evaluate_one(task)
                results.append(result)
                _print_progress(result, result["topology_idx"], params)
        else:
            # chunksize=1: a worker never holds more than one topology's matrix
            # at a time; map() yields in submission order so output stays ordered.
            with ProcessPoolExecutor(**pool_kwargs) as ex:
                results = []
                for result in ex.map(_evaluate_one, tasks, chunksize=1):
                    results.append(result)
                    _print_progress(result, result["topology_idx"], params)

        all_results[key] = results
        out_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nExperiment saved → {exp_dir}/")
    print(f"  equilibriums.json  ({out_path.stat().st_size:,} bytes)")
    print(f"  config.yaml")
    print(f"  topologies.json    ({topo_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
