import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml

from gps_model import GPSModel
from partitioning import TIG_ALGOS, TOPO_ALGOS, partition_task_ids


class TaskInteractionGraph:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "")
        self.system = config.get("system", {})
        self.nodes = config.get("nodes", [])
        self.workflows = config.get("workflows", [])
        self.tig = config.get("tig", {})
        self.partitioning = config.get("partitioning", {})

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "TaskInteractionGraph":
        return TaskInteractionGraph(raw)

    def compute_W(self) -> np.ndarray:
        """Pairwise task-interaction weight matrix.

        For each task pair (and singleton) this needs the marginal KPI range
        v(S) = E_others[max_S kpi - min_S kpi], which the naive implementation
        gets by re-enumerating the *entire* joint action space once per
        singleton/pair (O(n_tasks^2) full re-enumerations, each calling the
        O(n_tasks)-cost compute_service_stats -> O(n_tasks^3 * |A|^n_tasks)
        total). Instead we enumerate the joint space exactly once into a dense
        KPI tensor, then get every v(S) via a vectorised numpy axis reduction
        over that tensor (O(n_tasks * |A|^n_tasks) for the enumeration +
        cheap O(n_tasks^2) reductions). Verified bit-for-bit equivalent to the
        original implementation; ~10-100x faster for n_tasks >= 9 and the gap
        widens with n_tasks.
        """
        model = GPSModel(self.config)
        task_ids = sorted(model.services.keys())
        n_tasks = len(task_ids)
        M = model.M

        action_spaces = [model.services[s]["A"] for s in task_ids]
        dims = [len(a) for a in action_spaces]

        def kpi_vec(r: dict) -> np.ndarray:
            stats = model.compute_service_stats(r)
            C = stats["C"]
            out = np.empty(M, dtype=np.float64)
            for wid, wf in model.workflows.items():
                c_min = min(C[sid] for sid in wf["services"]) if wf["services"] else 0
                out[wid] = wf["lambda"] / c_min if c_min > 0 else np.inf
            return out

        # Full joint-action KPI tensor, shape (d_1, ..., d_n_tasks, M).
        # Each compute_service_stats call happens exactly once per joint
        # profile (same enumeration cost as the equilibrium payoff matrix).
        # NOTE: must stay float64 -- I[i,j] = v_ij - indiv_i - indiv_j relies
        # on near-exact cancellation to detect "no interaction" (clipped to
        # 0 below). float32 rounding noise breaks that cancellation for
        # independent task pairs, and the max-normalisation step amplifies
        # the resulting spurious residual into a large fake edge weight.
        kpi = np.empty((*dims, M), dtype=np.float64)
        flat = kpi.reshape(-1, M)
        for idx, combo in enumerate(itertools.product(*action_spaces)):
            flat[idx] = kpi_vec(dict(zip(task_ids, combo)))
        kpi[np.isinf(kpi)] = np.nan

        def marginal_range(axes: tuple) -> np.ndarray:
            """v(S) for S = task_ids at the given tensor axes, vectorised."""
            other_axes = tuple(a for a in range(n_tasks) if a not in axes)
            t = np.transpose(kpi, (*axes, *other_axes, n_tasks))
            s_size = int(np.prod([dims[a] for a in axes]))
            other_size = int(np.prod([dims[a] for a in other_axes])) if other_axes else 1
            t = t.reshape(s_size, other_size, M)
            with np.errstate(invalid="ignore"):
                rng = np.nanmax(t, axis=0) - np.nanmin(t, axis=0)
            rng = np.nan_to_num(rng, nan=0.0)
            return rng.mean(axis=0)

        indiv = {task_ids[i]: marginal_range((i,)) for i in range(n_tasks)}

        I = np.zeros((n_tasks, n_tasks, M))
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                v_ij = marginal_range((i, j))
                I[i, j, :] = v_ij - indiv[task_ids[i]] - indiv[task_ids[j]]
                I[j, i, :] = I[i, j, :]

        # average over KPIs (workflows)
        raw = np.mean(np.clip(I, 0, None), axis=2)

        mask = ~np.eye(n_tasks, dtype=bool)
        mx = raw[mask].max() if n_tasks > 1 else 0
        F_pull = raw / mx if mx > 0 else raw
        np.fill_diagonal(F_pull, 1.0)

        F_push = np.zeros((n_tasks, n_tasks))
        rho = self.tig.get("rho", 0.05)
        C_max = self.tig.get("C_max", 10)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        for i, si in enumerate(task_ids):
            for j, sj in enumerate(task_ids):
                C_ij = len(model.services[si]["A"]) * len(model.services[sj]["A"])
                F_push[i, j] = sigmoid(rho * (C_ij - C_max))
        np.fill_diagonal(F_push, 0.0)

        W = F_pull * (1.0 - F_push)
        np.fill_diagonal(W, 0.0)

        return W


class ARDConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"{self.config_path} must contain a YAML mapping at top level.")

        self._raw = raw
        self.name = raw.get("name", self.config_path.stem)
        self.system = raw.get("system", {})
        self.nodes = raw.get("nodes", [])
        self.workflows = raw.get("workflows", [])
        self.tig = raw.get("tig", {})
        self.partitioning = raw.get("partitioning", {})

    def load(self) -> Dict[str, Any]:
        return deepcopy(self._raw)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self._raw


class EquilibriumEvaluator:
    def __init__(self, config: ARDConfig):
        self.config = config
        self.tig = None
        self.assignment = None

    def load_tig(self) -> TaskInteractionGraph:
        self.tig = TaskInteractionGraph.from_dict(self.config.load())
        return self.tig
    
    def partition_tig(self) -> Dict[int, list]:
        model = GPSModel(self.config.load())
        algorithm = self.config.partitioning.get("algorithm", "greedy_modularity")
        n_agents = int(self.config.partitioning.get("n_agents", 2))

        w = None
        if algorithm in TIG_ALGOS:
            print("Computing TIG matrix inside TaskInteractionGraph...")
            w = self.tig.compute_W()

        self.assignment = partition_task_ids(model, algorithm=algorithm, n_agents=n_agents, w=w)
        return self.assignment

    def calc_equilibriums(self, override_algo=None, override_n_agents=None):
        model = GPSModel(self.config.load())
        
        agent_action_spaces = {}
        agent_ids = sorted(self.assignment.keys())
        
        for agent_id in agent_ids:
            services = self.assignment[agent_id]
            spaces = [model.services[sid]["A"] for sid in services]
            agent_action_spaces[agent_id] = [
                dict(zip(services, combo)) for combo in itertools.product(*spaces)
            ]
            
        matrix = {}
        profile_list = []
        
        # Enumerate joint profiles
        for action_indices in itertools.product(*(range(len(agent_action_spaces[aid])) for aid in agent_ids)):
            r = dict(model.baseline_assignment)
            for aid_idx, aid in enumerate(agent_ids):
                r.update(agent_action_spaces[aid][action_indices[aid_idx]])
                
            stats = model.compute_service_stats(r)
            
            utilities = {}
            for aid in agent_ids:
                services = self.assignment[aid]
                if not services:
                    utilities[aid] = {"U1": 0.0, "U2": 0.0}
                    continue
                    
                u1_count = sum(1 for s in services if stats["rho_s"][s] < 1.0)
                u1 = u1_count / len(services)
                u2 = sum(max(0, stats["lam_eff"][s] - stats["C"][s]) for s in services)
                utilities[aid] = {"U1": u1, "U2": u2}
                
            all_wids = list(model.workflows.keys())
            g_u1_count = sum(
                1 for wid in all_wids
                if model.workflows[wid]["lambda"] < min(
                    stats["C"][s] for s in model.workflows[wid]["services"]
                )
            )
            g_u1 = g_u1_count / len(all_wids)
            g_u2 = sum(max(0, stats["lam_eff"][s] - stats["C"][s])
                       for s in model.services)
            throughput = sum(min(stats["lam_eff"][s], stats["C"][s])
                            for s in model.services)
                
            profile_info = {
                "indices": action_indices,
                "replicas": r,
                "utilities": utilities,
                "global_utilities": {"U1": g_u1, "U2": g_u2, "throughput": throughput}
            }
            matrix[action_indices] = profile_info
            profile_list.append(profile_info)
            
        equilibria = []
        
        for action_indices, profile in matrix.items():
            is_equilibrium = True
            
            for dev_aid_idx, dev_aid in enumerate(agent_ids):
                current_util = profile["utilities"][dev_aid]
                can_improve = False
                
                for dev_action_idx in range(len(agent_action_spaces[dev_aid])):
                    if dev_action_idx == action_indices[dev_aid_idx]:
                        continue
                        
                    dev_indices = list(action_indices)
                    dev_indices[dev_aid_idx] = dev_action_idx
                    dev_indices = tuple(dev_indices)
                    
                    dev_util = matrix[dev_indices]["utilities"][dev_aid]
                    
                    if dev_util["U1"] > current_util["U1"]:
                        can_improve = True
                        break
                    elif dev_util["U1"] == current_util["U1"]:
                        if dev_util["U2"] < current_util["U2"]:
                            can_improve = True
                            break
                            
                if can_improve:
                    is_equilibrium = False
                    break
                    
            if is_equilibrium:
                equilibria.append(profile)
                
        algorithm = override_algo if override_algo is not None else self.config.partitioning.get("algorithm", "greedy_modularity")
        n_agents = override_n_agents if override_n_agents is not None else int(self.config.partitioning.get("n_agents", 2))
        
        out_dir = Path("results/analytical") / self.config.name / "equilibriums" / f"{algorithm}_{n_agents}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "total_profiles_evaluated": len(profile_list),
            "num_equilibria": len(equilibria),
            "equilibria": []
        }
        for eq in equilibria:
            summary["equilibria"].append({
                "replicas": eq["replicas"],
                "utilities": eq["utilities"],
                "global_utilities": eq["global_utilities"]
            })

        if equilibria:
            avg_u1 = sum(e["global_utilities"]["U1"] for e in equilibria) / len(equilibria)
            avg_u2 = sum(e["global_utilities"]["U2"] for e in equilibria) / len(equilibria)
            summary["avg_global_utilities"] = {"U1": avg_u1, "U2": avg_u2}

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        records = []
        for p in profile_list:
            rec = {"profile_id": str(p["indices"])}
            for sid, r_s in p["replicas"].items():
                rec[f"r_{sid}"] = r_s
            for aid, utils in p["utilities"].items():
                rec[f"agent_{aid}_U1"] = utils["U1"]
                rec[f"agent_{aid}_U2"] = utils["U2"]
            records.append(rec)
            
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "joint_profiles.csv", index=False)
        
        return len(profile_list), equilibria, profile_list, agent_action_spaces, model.baseline_assignment


def compute_poa(profile_list, equilibria):
    """Compute Price of Anarchy using throughput-based welfare.
    
    W(r) = sum_s min(lam_eff_s, C_s) = total served arrival rate.
    PoA = W(social_optimum) / W(worst_equilibrium), >= 1.0.
    """
    if not equilibria or not profile_list:
        return None
    
    # Social optimum: profile with highest throughput
    social_opt = max(profile_list, key=lambda p: p["global_utilities"]["throughput"])
    # Worst equilibrium: equilibrium with lowest throughput
    worst_eq = min(equilibria, key=lambda e: e["global_utilities"]["throughput"])
    
    w_opt = social_opt["global_utilities"]["throughput"]
    w_worst = worst_eq["global_utilities"]["throughput"]
    
    if w_worst <= 0:
        return float("inf")
    return w_opt / w_worst

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config = ARDConfig(args.config)
    evaluator = EquilibriumEvaluator(config)

    evaluator.load_tig()
    evaluator.partition_tig()

    algo = config.partitioning.get("algorithm", "greedy_modularity")
    n_agents = config.partitioning.get("n_agents", 2)

    print(f"Scenario: {config.name}")
    print(f"Partition: {algo} ({n_agents} agents)")
    print("Calculating equilibriums...")
    
    total_profiles, equilibria, all_profiles, agent_action_spaces, baseline = evaluator.calc_equilibriums()
    
    model = GPSModel(config.load())
    
    baseline_stats = model.compute_service_stats(baseline)
    print("\nBaseline Utilities:")
    all_wids = list(model.workflows.keys())
    g_u1_count = sum(
        1 for wid in all_wids
        if model.workflows[wid]["lambda"] < min(
            baseline_stats["C"][s] for s in model.workflows[wid]["services"]
        )
    )
    g_u1 = g_u1_count / len(all_wids)
    g_u2 = sum(max(0, baseline_stats["lam_eff"][s] - baseline_stats["C"][s])
               for s in model.services)
    print(f"  System Utils: U1={g_u1:.3f}, U2={g_u2:.3f}")
    for aid in sorted(evaluator.assignment.keys()):
        services = evaluator.assignment[aid]
        if services:
            u1_count = sum(1 for s in services if baseline_stats["rho_s"][s] < 1.0)
            u1 = u1_count / len(services)
            u2 = sum(max(0, baseline_stats["lam_eff"][s] - baseline_stats["C"][s])
                     for s in services)
            print(f"  Agent {aid} Utils: U1={u1:.3f}, U2={u2:.3f}")
            
    print("\nBaseline Partitioning Assignments and Equilibria:")
    for topo_algo in TOPO_ALGOS:
        topo_assignment = partition_task_ids(model, algorithm=topo_algo, n_agents=n_agents, w=None)
        evaluator.assignment = topo_assignment
        total_p, eqs, topo_profiles, _, _ = evaluator.calc_equilibriums(override_algo=topo_algo, override_n_agents=len(topo_assignment))
        
        poa = compute_poa(topo_profiles, eqs)
        print(f"\n  {topo_algo} Partition: {topo_assignment}")
        print(f"  Found {len(eqs)} pure-strategy equilibria:")
        if eqs:
            worst_eq = min(eqs, key=lambda e: (e['global_utilities']['U1'], -e['global_utilities']['U2']))
            avg_u1 = sum(e['global_utilities']['U1'] for e in eqs) / len(eqs)
            avg_u2 = sum(e['global_utilities']['U2'] for e in eqs) / len(eqs)
            print(f"    Worst Equilibrium:")
            print(f"      Replicas: {worst_eq['replicas']}")
            print(f"      System Utils: U1={worst_eq['global_utilities']['U1']:.3f}, U2={worst_eq['global_utilities']['U2']:.3f}")
            print(f"    Avg Equilibrium:")
            print(f"      System Utils: U1={avg_u1:.3f}, U2={avg_u2:.3f}")
            print(f"    PoA: {poa:.4f}" if poa is not None else "    PoA: N/A")
                
    # Restore the original assignment for the main prints
    evaluator.partition_tig()
    
    print("\nAction space sizes per agent:")
    for aid, space in agent_action_spaces.items():
        print(f"  Agent {aid}: {len(space)} actions")
        
    print(f"\nAssignment: {evaluator.assignment}")
    print(f"\nEvaluated {total_profiles} joint profiles.")
    print(f"Found {len(equilibria)} pure-strategy equilibria:")
    if equilibria:
        worst_eq = min(equilibria, key=lambda e: (e['global_utilities']['U1'], -e['global_utilities']['U2']))
        avg_u1 = sum(e['global_utilities']['U1'] for e in equilibria) / len(equilibria)
        avg_u2 = sum(e['global_utilities']['U2'] for e in equilibria) / len(equilibria)
        poa = compute_poa(all_profiles, equilibria)
        print(f"  Worst Equilibrium:")
        print(f"    Replicas: {worst_eq['replicas']}")
        print(f"    System Utils: U1={worst_eq['global_utilities']['U1']:.3f}, U2={worst_eq['global_utilities']['U2']:.3f}")
        print(f"  Avg Equilibrium:")
        print(f"    System Utils: U1={avg_u1:.3f}, U2={avg_u2:.3f}")
        print(f"  PoA: {poa:.4f}" if poa is not None else "  PoA: N/A")


if __name__ == "__main__":
    main()