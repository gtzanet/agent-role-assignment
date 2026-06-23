"""Generate service-to-node topology mappings for M ∈ {m_min..m_max} × N ∈ {n_min..n_max}.

Each workflow consists of two services, so M workflows → 2M services.
A mapping assigns each service to one of N nodes.

When N^(2M) > max_topologies, a random sample is drawn instead of enumerating
all mappings so that the dataset stays tractable for downstream evaluation.

Usage:
    python3 equilibrium_analysis/generate_topology_dataset.py
    python3 equilibrium_analysis/generate_topology_dataset.py --m 1 5 --n 1 3
    python3 equilibrium_analysis/generate_topology_dataset.py --m 1 5 --n 1 5 --max-topologies 200 --seed 7

Output: equilibrium_analysis/topologies/topologies_M<m_min>-<m_max>_N<n_min>-<n_max>_cap<max_topologies>_seed<seed>.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def generate_mappings(M: int, N: int, max_topologies: int, seed: int, services_per_wf: int = 2) -> dict:
    num_services = services_per_wf * M
    services     = [f"w{w}_s{s}" for w in range(M) for s in range(services_per_wf)]
    nodes        = [f"n{j}" for j in range(N)]
    total_topos  = N ** num_services

    if total_topos <= max_topologies:
        mappings = [
            {services[i]: assignment[i] for i in range(num_services)}
            for assignment in itertools.product(nodes, repeat=num_services)
        ]
        sampled = False
    else:
        rng  = random.Random(seed)
        seen: set[tuple] = set()
        mappings = []
        while len(mappings) < max_topologies:
            assignment = tuple(rng.choice(nodes) for _ in range(num_services))
            if assignment not in seen:
                seen.add(assignment)
                mappings.append(
                    {services[i]: assignment[i] for i in range(num_services)}
                )
        sampled = True

    return {
        "M":                M,
        "N":                N,
        "services_per_wf":  services_per_wf,
        "num_services":     num_services,
        "num_nodes":        N,
        "services":         services,
        "nodes":            nodes,
        "total_topologies": total_topos,
        "num_mappings":     len(mappings),
        "sampled":          sampled,
        "mappings":         mappings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate service-to-node topology datasets."
    )
    parser.add_argument(
        "--m", nargs=2, type=int, metavar=("MIN", "MAX"), default=[1, 5],
        help="Range of M (workflows). Default: 1 5",
    )
    parser.add_argument(
        "--n", nargs=2, type=int, metavar=("MIN", "MAX"), default=[1, 5],
        help="Range of N (nodes). Default: 1 5",
    )
    parser.add_argument(
        "--max-topologies", type=int, default=100, metavar="CAP",
        help="Max topologies per (M,N) pair before random sampling. Default: 100",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sampling. Default: 42",
    )
    parser.add_argument(
        "--services-per-wf", type=int, default=2, metavar="S",
        help="Number of services per workflow. Default: 2",
    )
    args = parser.parse_args()

    m_min, m_max    = args.m
    n_min, n_max    = args.n
    max_topos       = args.max_topologies
    seed            = args.seed
    services_per_wf = args.services_per_wf

    dataset: dict = {}
    for M in range(m_min, m_max + 1):
        for N in range(n_min, n_max + 1):
            key  = f"M{M}_N{N}"
            data = generate_mappings(M, N, max_topos, seed, services_per_wf)
            dataset[key] = data
            tag = (f"(sampled {data['num_mappings']}/{data['total_topologies']:,})"
                   if data["sampled"] else "")
            print(f"{key}: {data['num_mappings']:>4} mappings  "
                  f"services={data['services']}  nodes={data['nodes']}  {tag}")

    out_dir = _SCRIPT_DIR / "topologies"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = (f"topologies"
             f"_M{m_min}-{m_max}"
             f"_N{n_min}-{n_max}"
             f"_S{services_per_wf}"
             f"_cap{max_topos}"
             f"_seed{seed}"
             f".json")
    out_path = out_dir / fname
    out_path.write_text(json.dumps(dataset, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
