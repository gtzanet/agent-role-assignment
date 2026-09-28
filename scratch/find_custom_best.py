import itertools
import sys
import yaml

sys.path.append('.')
from evaluate_equilibriums import EquilibriumEvaluator, ARDConfig, GPSModel

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
                    {"id": 1, "node": 1, "replicas_baseline": 1, "action_space": [1, 2]}
                ]
            },
            {
                "id": 1, "lambda": 35.0,
                "services": [
                    {"id": 2, "node": 1, "replicas_baseline": 1, "action_space": [1, 2]},
                    {"id": 3, "node": 0, "replicas_baseline": 1, "action_space": [1, 2]}
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

def get_all_partitions():
    return [
        ("all_in_one", {0: [0, 1, 2, 3]}),
        ("all_separate", {0: [0], 1: [1], 2: [2], 3: [3]}),
        ("per_workflow", {0: [0, 1], 1: [2, 3]}),
        ("per_node", {0: [0, 3], 1: [1, 2]}),
        
        # 2-agent custom partitions
        ("custom_diagonal_1", {0: [0, 2], 1: [1, 3]}),
        ("custom_3on1_A", {0: [0, 1, 3], 1: [2]}),
        ("custom_3on1_B", {0: [0, 2, 3], 1: [1]}),
        ("custom_3on1_C", {0: [1, 2, 3], 1: [0]}),
        ("custom_3on1_D", {0: [0, 1, 2], 1: [3]}),
        
        # 3-agent custom partitions
        ("custom_3A_1", {0: [0, 1], 1: [2], 2: [3]}),
        ("custom_3A_2", {0: [0, 2], 1: [1], 2: [3]}),
        ("custom_3A_3", {0: [0, 3], 1: [1], 2: [2]}),
        ("custom_3A_4", {0: [1, 2], 1: [0], 2: [3]}),
        ("custom_3A_5", {0: [1, 3], 1: [0], 2: [2]}),
        ("custom_3A_6", {0: [2, 3], 1: [0], 2: [1]}),
    ]

def evaluate_params(lam0, lam1, delta, f0):
    cfg_dict = create_base_config()
    cfg_dict["workflows"][0]["lambda"] = lam0
    cfg_dict["workflows"][1]["lambda"] = lam1
    cfg_dict["delta"] = delta
    cfg_dict["nodes"][0]["f"] = f0
    cfg_dict["nodes"][1]["f"] = f0

    with open("scratch/temp_cfg2.yaml", "w") as f:
        yaml.dump(cfg_dict, f)

    config = ARDConfig("scratch/temp_cfg2.yaml")
    evaluator = EquilibriumEvaluator(config)
    evaluator.assignment = {}
    
    results = {}
    for name, assign in get_all_partitions():
        evaluator.assignment = assign
        _, eqs, _, _, _ = evaluator.calc_equilibriums(override_algo=name, override_n_agents=len(assign))
        if eqs:
            worst_eq = min(eqs, key=lambda e: (e['global_utilities']['U1'], -e['global_utilities']['U2']))
            results[name] = worst_eq['global_utilities']
        else:
            results[name] = {"U1": -1, "U2": 999999}
            
    return results

def main():
    lam_vals = [30]
    lam1_vals = [10]
    delta_vals = [0]
    f_vals = [350]
    
    decentralized_baselines = ["all_separate", "per_workflow", "per_node"]
    baselines = ["all_in_one"] + decentralized_baselines
    
    print("Searching for config where custom partition beats all decentralized baselines in U1...")
    for lam0, lam1, delta, f in itertools.product(lam_vals, lam1_vals, delta_vals, f_vals):
        res = evaluate_params(lam0, lam1, delta, f)
        
        dec_max_u1 = max(res[b]["U1"] for b in decentralized_baselines)
        aio_u1 = res["all_in_one"]["U1"]
        
        print(f"Results for lam0={lam0}, lam1={lam1}, delta={delta}, f={f}")
        for name, utils in res.items():
            print(f"{name}: U1={utils['U1']:.3f}, U2={utils['U2']:.3f}")

if __name__ == '__main__':
    main()
