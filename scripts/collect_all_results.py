#!/usr/bin/env python3
"""Collect analytical results into a single CSV.

Writes: results/analytical/all_results.csv

Columns include policy metrics plus partition-alignment scores:
scenario, partition, overall_drop_pct, mean_node_util_pct, workflow_cohesion_pct,
node_cohesion_pct, n_agents, additional_json...
"""
import csv
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "analytical"
OUT_CSV = RESULTS_DIR / "all_results.csv"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _build_service_index(partition_summary: dict) -> dict[int, dict]:
    services: dict[int, dict] = {}
    for agent in partition_summary.get("agents", []):
        service_ids = agent.get("service_ids", [])
        nodes = agent.get("nodes", [])
        workflows = agent.get("workflows", [])
        for idx, service_id in enumerate(service_ids):
            services[int(service_id)] = {
                "partition": int(agent.get("agent_id", 0)),
                "node": nodes[idx] if idx < len(nodes) else None,
                "workflow": workflows[idx] if idx < len(workflows) else None,
            }
    return services


def _pairwise_cohesion_pct(items: list[dict], key: str) -> Optional[float]:
    total = 0
    same = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i].get(key) is None or items[j].get(key) is None:
                continue
            if items[i][key] == items[j][key]:
                total += 1
                if items[i]["partition"] == items[j]["partition"]:
                    same += 1
    if total == 0:
        return None
    return round(100.0 * same / total, 3)


def _partition_alignment_metrics(partition_summary: dict) -> dict:
    service_index = _build_service_index(partition_summary)
    services = [
        {"service_id": sid, **meta}
        for sid, meta in sorted(service_index.items(), key=lambda item: item[0])
    ]
    return {
        "workflow_cohesion_pct": _pairwise_cohesion_pct(services, "workflow"),
        "node_cohesion_pct": _pairwise_cohesion_pct(services, "node"),
    }


def collect():
    rows = []
    for scenario_dir in sorted(RESULTS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        policy_root = scenario_dir / "policy"
        if not policy_root.exists():
            continue
        for partition_dir in sorted(policy_root.iterdir()):
            if not partition_dir.is_dir():
                continue
            partition_summary = _load_json(
                scenario_dir / "partitions" / partition_dir.name / "summary.json"
            )
            ms = partition_dir / "metrics_summary.json"
            if not ms.exists():
                # maybe nested under final_metrics or summary
                ms_alt = partition_dir / "summary.json"
                ms = ms_alt if ms_alt.exists() else None
            if ms is None:
                continue
            data = _load_json(ms)
            if data is None:
                continue
            alignment = _partition_alignment_metrics(partition_summary) if partition_summary else {
                "workflow_cohesion_pct": None,
                "node_cohesion_pct": None,
            }
            row = {
                "scenario": scenario_dir.name,
                "partition": partition_dir.name,
                "overall_drop_pct": data.get("overall_drop_pct"),
                "mean_node_util_pct": data.get("mean_node_util_pct"),
                "workflow_cohesion_pct": alignment["workflow_cohesion_pct"],
                "node_cohesion_pct": alignment["node_cohesion_pct"],
                "n_agents": data.get("n_agents") or data.get("learned_agent_count") or "",
            }
            # include other keys as json blob
            extra = {k: v for k, v in data.items() if k not in row}
            row["extra_json"] = json.dumps(extra)
            rows.append(row)

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "partition",
        "overall_drop_pct",
        "mean_node_util_pct",
        "workflow_cohesion_pct",
        "node_cohesion_pct",
        "n_agents",
        "extra_json",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    collect()
    print(f"Wrote {OUT_CSV}")
