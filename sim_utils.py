from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _count_workflows(topology_cfg: dict[str, Any]) -> int:
    workflows = topology_cfg.get("workflows")
    if isinstance(workflows, list):
        return len(workflows)

    n_workflows = topology_cfg.get("n_workflows")
    if n_workflows is not None:
        return int(n_workflows)

    return 0


class MetricsCollector:
    """Captures eval-window metrics and optionally perturbs thread counts."""

    def __init__(self, perturb_threads: bool, perturb_prob: float, thread_min: int, thread_max: int):
        self.windows: list[tuple[dict, dict]] = []
        self.perturb_threads = bool(perturb_threads)
        self.perturb_prob = float(np.clip(perturb_prob, 0.0, 1.0))
        self.thread_min = int(max(1, thread_min))
        self.thread_max = int(max(self.thread_min, thread_max))

    def on_eval(self, idx, _service, accumulated, instant):
        if idx == 0:
            self.windows.append((accumulated, instant))

        if not self.perturb_threads:
            return None

        if np.random.random() > self.perturb_prob:
            return None

        new_threads = int(np.random.randint(self.thread_min, self.thread_max + 1))
        return {"cpu": new_threads}

    def reset(self):
        self.windows.clear()


def build_dataset_row(collector: MetricsCollector, nodes: list[Any], cpu_max: int, n_services: int, n_workflows: int) -> dict[str, float]:
    """Aggregate one simulation run into a single dataset row."""
    row: dict[str, float] = {}

    for sid in range(n_services):
        latencies = [
            w[0]["services"][sid]["avg_latency"]
            for w in collector.windows
            if w[0]["services"][sid]["arrivals"] > 0
        ]
        throughputs = [
            w[0]["services"][sid]["avg_throughput"]
            for w in collector.windows
            if w[0]["services"][sid]["arrivals"] > 0
        ]
        row[f"s{sid}_avg_latency"] = _safe_mean(latencies)
        row[f"s{sid}_arrival_rate"] = _safe_mean([w[0]["services"][sid]["arrival_rate"] for w in collector.windows])
        row[f"s{sid}_departure_rate"] = _safe_mean([w[0]["services"][sid]["departure_rate"] for w in collector.windows])
        row[f"s{sid}_avg_throughput"] = _safe_mean(throughputs)
        row[f"s{sid}_avg_queue_size"] = _safe_mean([w[1]["services"][sid]["queue_size"] for w in collector.windows])
        row[f"s{sid}_avg_threads"] = _safe_mean([w[1]["services"][sid]["threads"] for w in collector.windows])

    for wid in range(n_workflows):
        e2e_lats = [
            w[0]["workflows"]["e2e_latencies"][wid]
            for w in collector.windows
            if w[0]["workflows"]["e2e_latencies"][wid] > 0
        ]
        row[f"wf{wid}_avg_e2e_latency"] = _safe_mean(e2e_lats)
        row[f"wf{wid}_violation_rate"] = _safe_mean([w[0]["workflows"]["violation_rates"][wid] for w in collector.windows])

    for node in nodes:
        cpu_arr = np.array(node.cpu_metric, dtype=float)
        avg_threads = float(np.mean(cpu_arr)) if cpu_arr.size > 0 else 0.0
        row[f"node{node.id}_avg_cpu_threads"] = avg_threads
        row[f"node{node.id}_cpu_usage_pct"] = min(100.0, 100.0 * avg_threads / max(float(cpu_max), 1e-12))

    return row


def resolve_sim_config_path(config_arg: str, workflow_sim_root: Path) -> Path:
    sim_config_path = Path(config_arg)
    if not sim_config_path.is_absolute():
        sim_config_path = workflow_sim_root / sim_config_path
    if not sim_config_path.exists():
        raise FileNotFoundError(f"Simulator config file not found: {sim_config_path}")
    return sim_config_path


def load_simulator_config(sim_config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(sim_config_path.read_text()) or {}
    topology_cfg = config.get("topology", {})
    for wf in topology_cfg.get("workflows", []):
        wf["edges"] = [tuple(e) for e in wf.get("edges", [])]
    return config


def reset_app_threads(app: Any, thread_count: int = 1) -> None:
    app.reset()
    for service in app.services:
        service.threads = thread_count


@dataclass
class SimulatorRuntime:
    app: Any
    nodes: list[Any]
    agents: dict[int, MetricsCollector]
    collector: MetricsCollector
    n_services: int
    n_workflows: int


def build_runtime(
    config: dict[str, Any],
    simulation_classes: tuple[type, type],
    perturb_threads: bool,
    perturb_prob: float,
    thread_min: int,
    thread_max: int,
) -> SimulatorRuntime:
    application_cls, sim_node_cls = simulation_classes

    topology_cfg = config.get("topology", {})
    infra_cfg = config.get("infrastructure", {})

    n_nodes = int(infra_cfg.get("n_nodes", 3))
    cpu_max = int(infra_cfg.get("cpu_max", 4))
    ram = int(infra_cfg.get("ram", 8))
    freq = int(infra_cfg.get("freq", 1000))

    app = application_cls(topology=topology_cfg)
    n_services = len(app.services)
    n_workflows = _count_workflows(topology_cfg)

    nodes = [sim_node_cls(i, cpu_max, ram, freq) for i in range(n_nodes)]
    initial_service_map = {service.id: nodes[service.id % n_nodes] for service in app.services}
    app.deploy_services(initial_service_map)
    reset_app_threads(app, thread_count=1)

    collector = MetricsCollector(
        perturb_threads=perturb_threads,
        perturb_prob=perturb_prob,
        thread_min=thread_min,
        thread_max=thread_max,
    )
    agents = {service.id: collector for service in app.services}

    return SimulatorRuntime(
        app=app,
        nodes=nodes,
        agents=agents,
        collector=collector,
        n_services=n_services,
        n_workflows=n_workflows,
    )


def run_single_sample(
    simulation_cls: type,
    runtime: SimulatorRuntime,
    iterations: int,
    timeout: int,
    eval_interval: float,
    latency_target: Any,
) -> None:
    reset_app_threads(runtime.app, thread_count=1)
    runtime.collector.reset()

    sim = simulation_cls(
        apps=[runtime.app],
        units=[],
        iterations=iterations,
        timeout=timeout,
        eval_interval=eval_interval,
        latency_target=latency_target,
    )
    sim.run(agents=runtime.agents)


def save_rows_to_csv(rows: list[dict[str, float]], output_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df