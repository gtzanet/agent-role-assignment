import itertools
import yaml
from copy import deepcopy
from pathlib import Path
import sys

# Add parent to path to import evaluator
sys.path.append('.')
from evaluate_equilibriums import EquilibriumEvaluator, ARDConfig, partition_task_ids, GPSModel, compute_poa

def create_base_config():
    return {
        "name": "search",
        "system": {"N": 2, "M": 2, "L": 10},
        "nodes": [
            {"id": 0, "f": 1000},
            {"id": 1, "f": 1000}
        ],
        "delta": 5,
        "workflows": [
            {
                "id": 0, "lambda": 35.0,
                "services": [
                    {"id": 0, "node": 0, "replicas_baseline": 1, "action_space": [1, 2]},
                    {"id": 1, "node": 0, "replicas_baseline": 1, "action_space": [1, 2]}
                ]
            },
            {
                "id": 1, "lambda": 35.0,
                "services": [
                    {"id": 2, "node": 0, "replicas_baseline": 1, "action_space": [1, 2]},
                    {"id": 3, "node": 1, "replicas_baseline": 1, "action_space": [1, 2]}
                ]
            }
        ],
        "tig": {
            "C_max": 100,
            "rho": 0.1,
            "kpi_weights": {"drop_rate": 0.5, "utilization": 0.5},
            "delta_method": "sii_marginalised"
        },
        "partitioning": {"algorithm": "per_workflow", "n_agents": 2}
    }

def evaluate_params(lam0, lam1, delta, f0):
    cfg_dict = create_base_config()
    cfg_dict["workflows"][0]["lambda"] = lam0
    cfg_dict["workflows"][1]["lambda"] = lam1
    cfg_dict["delta"] = delta
    cfg_dict["nodes"][0]["f"] = f0
    cfg_dict["nodes"][1]["f"] = f0

    with open("scratch/temp_cfg.yaml", "w") as f:
        yaml.dump(cfg_dict, f)

    config = ARDConfig("scratch/temp_cfg.yaml")
    evaluator = EquilibriumEvaluator(config)
    evaluator.assignment = {}  # Prevent crash if tig is missing, we override anyway
    
    model = GPSModel(config.load())
    
    results = {}
    partitions = {
        "per_workflow": {0: [0, 1], 1: [2, 3]},
        "per_node": {0: [0, 1, 2], 1: [3]},
        "all_separate": {0: [0], 1: [1], 2: [2], 3: [3]}
    }
    
    for name, assign in partitions.items():
        evaluator.assignment = assign
        _, eqs, all_profiles, _, _ = evaluator.calc_equilibriums(override_algo=name, override_n_agents=len(assign))
        if eqs:
            # We want worst eq U1 and U2
            worst_eq = min(eqs, key=lambda e: (e['global_utilities']['U1'], -e['global_utilities']['U2']))
            results[name] = worst_eq['global_utilities']
        else:
            results[name] = {"U1": -1, "U2": 999999}
            
    return results

def main():
    lam_vals = [10, 15, 20, 25, 30, 35, 40]
    delta_vals = [0, 5, 10, 15, 20, 25]
    f_vals = [300, 400, 500, 600, 800, 1000]
    
    found_case_a = False # per_node > per_workflow
    found_case_b = False # per_workflow > per_node
    found_case_c = False # all_separate > both
    
    print("Searching...")
    for lam0, lam1, delta, f in itertools.product(lam_vals, lam_vals, delta_vals, f_vals):
        res = evaluate_params(lam0, lam1, delta, f)
        
        def is_better(u_a, u_b):
            # Returns true if u_a is strictly better than u_b
            if u_a["U1"] > u_b["U1"]: return True
            if u_a["U1"] == u_b["U1"] and u_a["U2"] < u_b["U2"]: return True
            return False
            
        pw = res["per_workflow"]
        pn = res["per_node"]
        al = res["all_separate"]
        
        if is_better(pn, pw) and not found_case_a:
            print(f"CASE A (per_node > per_workflow): lam0={lam0}, lam1={lam1}, delta={delta}, f={f}")
            print(f"  per_node:     U1={pn['U1']:.3f}, U2={pn['U2']:.3f}")
            print(f"  per_workflow: U1={pw['U1']:.3f}, U2={pw['U2']:.3f}")
            print(f"  all_separate: U1={al['U1']:.3f}, U2={al['U2']:.3f}")
            found_case_a = True
            
        if is_better(pw, pn) and not found_case_b:
            print(f"CASE B (per_workflow > per_node): lam0={lam0}, lam1={lam1}, delta={delta}, f={f}")
            print(f"  per_workflow: U1={pw['U1']:.3f}, U2={pw['U2']:.3f}")
            print(f"  per_node:     U1={pn['U1']:.3f}, U2={pn['U2']:.3f}")
            print(f"  all_separate: U1={al['U1']:.3f}, U2={al['U2']:.3f}")
            found_case_b = True
            
        if is_better(al, pw) and is_better(al, pn) and not found_case_c:
            print(f"CASE C (all_separate > both): lam0={lam0}, lam1={lam1}, delta={delta}, f={f}")
            print(f"  all_separate: U1={al['U1']:.3f}, U2={al['U2']:.3f}")
            print(f"  per_workflow: U1={pw['U1']:.3f}, U2={pw['U2']:.3f}")
            print(f"  per_node:     U1={pn['U1']:.3f}, U2={pn['U2']:.3f}")
            found_case_c = True

if __name__ == '__main__':
    main()
