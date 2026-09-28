"""
Exhaustive partition evaluator for the analytical GPS model.

The script loads a partition assignment, enumerates the joint action space of
each group, and evaluates the system metrics for every joint action while
keeping all non-group services at their baseline replica counts.

For each group, it reports:
  - the joint action space size
  - the worst mean node utilisation across all joint actions
  - the worst mean drop rate across all joint actions

It also prints the group with the largest joint action space.

Usage:
    python evaluate_partition_joint_actions.py \
        --partition spectral_sii_marginalised \
        [--config analytical_config.yaml]

    python evaluate_partition_joint_actions.py \
        --assignment-file results/analytical/<scenario>/partitions/<label>/assignment.json \
        [--config analytical_config.yaml]
"""

import argparse
import json
from dataclasses import dataclass
from itertools import product as cart_product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from compute_analytical_metrics import GPSModel


@dataclass
class GroupEvaluation:
    agent_id: int
    service_ids: List[int]
    action_sizes: List[int]
    joint_space: int
    worst_mean_node_util: float
    worst_mean_drop_rate: float


def load_assignment(partition_label: Optional[str], assignment_file: Optional[str], scenario_dir: Path) -> Tuple[Dict[int, List[int]], Path]:
    if assignment_file is not None:
        path = Path(assignment_file)
    elif partition_label is not None:
        path = scenario_dir / "partitions" / partition_label / "assignment.json"
    else:
        raise ValueError("Either --partition or --assignment-file must be provided")

    if not path.exists():
        raise FileNotFoundError(f"Partition assignment not found: {path}")

    with open(path) as fh:
        raw = json.load(fh)

    assignment = {int(agent_id): [int(service_id) for service_id in service_ids]
                  for agent_id, service_ids in raw.items()}
    return assignment, path


def mean_node_util(kpis: dict, model: GPSModel) -> float:
    return float(np.mean([kpis[f"u_node{n}"] for n in range(model.N)]))


def mean_drop_rate(kpis: dict, model: GPSModel) -> float:
    return float(np.mean([kpis[f"D_wf{w}/λ"] for w in range(model.M)]))


def action_space_string(action_sizes: Iterable[int]) -> str:
    action_sizes = list(action_sizes)
    if not action_sizes:
        return "1 = 1 combination"

    total = 1
    for size in action_sizes:
        total *= size
    return f"{'×'.join(str(size) for size in action_sizes)} = {total:,} combinations"


def evaluate_group(model: GPSModel, assignment: Dict[int, List[int]], agent_id: int) -> GroupEvaluation:
    service_ids = assignment[agent_id]
    action_sizes = [len(model.services[sid]["A"]) for sid in service_ids]
    joint_space = 1
    for size in action_sizes:
        joint_space *= size

    baseline = dict(model.baseline_assignment)
    worst_mean_util = -np.inf
    worst_mean_drop = -np.inf

    action_spaces = [model.services[sid]["A"] for sid in service_ids]
    for combo in cart_product(*action_spaces):
        r = dict(baseline)
        r.update(dict(zip(service_ids, combo)))
        kpis = model.compute_kpis(r)
        worst_mean_util = max(worst_mean_util, mean_node_util(kpis, model))
        worst_mean_drop = max(worst_mean_drop, mean_drop_rate(kpis, model))

    return GroupEvaluation(
        agent_id=agent_id,
        service_ids=service_ids,
        action_sizes=action_sizes,
        joint_space=joint_space,
        worst_mean_node_util=float(worst_mean_util),
        worst_mean_drop_rate=float(worst_mean_drop),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="analytical_config.yaml")
    parser.add_argument("--partition", default=None,
                        help="Partition label under results/analytical/<scenario>/partitions/")
    parser.add_argument("--assignment-file", default=None,
                        help="Path to an assignment.json file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    scenario = cfg.get("name", config_path.stem)
    model = GPSModel(cfg)
    scenario_dir = Path("results/analytical") / scenario

    assignment, assignment_path = load_assignment(args.partition, args.assignment_file, scenario_dir)

    print(f"Config   : {config_path}")
    print(f"Scenario : {scenario}")
    print(f"Partition: {assignment_path}")
    print(f"Agents   : {len(assignment)}")

    evaluations: List[GroupEvaluation] = []
    for agent_id in sorted(assignment):
        evaluation = evaluate_group(model, assignment, agent_id)
        evaluations.append(evaluation)
        print(f"\nAgent {agent_id}")
        print(f"  Services : {evaluation.service_ids}")
        print(f"  Action space : {action_space_string(evaluation.action_sizes)}")
        print(f"  Worst mean node util : {evaluation.worst_mean_node_util:.3f}%")
        print(f"  Worst mean drop rate : {evaluation.worst_mean_drop_rate:.3f}%")

    largest_group = max(evaluations, key=lambda item: item.joint_space)
    worst_util_group = max(evaluations, key=lambda item: item.worst_mean_node_util)
    worst_drop_group = max(evaluations, key=lambda item: item.worst_mean_drop_rate)

    print(f"\n{'─' * 60}")
    print(f"Largest action space group: agent {largest_group.agent_id}")
    print(f"  Services : {largest_group.service_ids}")
    print(f"  Action space : {action_space_string(largest_group.action_sizes)}")
    print(f"Worst mean node util observed : {worst_util_group.worst_mean_node_util:.3f}% (agent {worst_util_group.agent_id})")
    print(f"Worst mean drop rate observed : {worst_drop_group.worst_mean_drop_rate:.3f}% (agent {worst_drop_group.agent_id})")


if __name__ == "__main__":
    main()