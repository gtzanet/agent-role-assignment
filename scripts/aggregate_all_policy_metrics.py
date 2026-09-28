import json
from pathlib import Path
from compute_analytical_metrics import GPSModel

SCENARIO = "M4N3_B_striped"
BASE_DIR = Path("results/analytical") / SCENARIO / "policy"
CFG_PATH = Path("analytical_config.yaml")

with open(CFG_PATH) as fh:
    cfg = json.loads(json.dumps(__import__('yaml').safe_load(fh)))
model = GPSModel(cfg)

summaries = []
for part_dir in sorted(BASE_DIR.iterdir()):
    if not part_dir.is_dir():
        continue
    assign_file = part_dir / "final_assignment.json"
    if not assign_file.exists():
        print(f"Skipping {part_dir}, no final_assignment.json")
        continue
    with open(assign_file) as fh:
        r = json.load(fh)
    r_num = {int(k[1:]): int(v) for k, v in r.items()}
    kpis = model.compute_kpis(r_num)
    lambdas = {w: model.workflows[w]['lambda'] for w in model.workflows}
    total_lambda = sum(lambdas.values())
    total_dropped = sum((kpis[f"D_wf{w}/λ"]/100.0) * lambdas[w]
                        for w in model.workflows)
    overall_drop_pct = total_dropped / total_lambda * 100.0
    mean_node_util = sum(kpis[f"u_node{n}"] for n in range(model.N)) / model.N
    summary = {
        "partition": part_dir.name,
        "overall_drop_pct": round(overall_drop_pct, 3),
        "mean_node_util_pct": round(mean_node_util, 3),
        "per_workflow_drop": {f"wf{w}": round(kpis[f"D_wf{w}/λ"],3) for w in model.workflows},
        "per_node_util": {f"n{n}": round(kpis[f"u_node{n}"],3) for n in range(model.N)},
    }
    with open(part_dir / "metrics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    summaries.append(summary)
    print(f"Wrote {part_dir}/metrics_summary.json")

with open(BASE_DIR / "aggregate_metrics_all.json", "w") as fh:
    json.dump(summaries, fh, indent=2)
print(f"Wrote aggregate_metrics_all.json")
