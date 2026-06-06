import json
from pathlib import Path
from compute_analytical_metrics import GPSModel

SCENARIO = "M4N3_B_striped"
PARTITIONS = [
    "spectral_sii_marginalised",
    "per_workflow",
    "per_node",
    "per_task",
]

BASE_DIR = Path("results/analytical") / SCENARIO / "policy"
CFG_PATH = Path("analytical_config.yaml")

with open(CFG_PATH) as fh:
    cfg = json.loads(json.dumps(__import__('yaml').safe_load(fh)))
model = GPSModel(cfg)

out_rows = []
for part in PARTITIONS:
    out_dir = BASE_DIR / part
    assign_file = out_dir / "final_assignment.json"
    if not assign_file.exists():
        print(f"Missing assignment for partition {part} at {assign_file}")
        continue
    with open(assign_file) as fh:
        r = json.load(fh)
    # r keys like 's0': 2 -> convert to {0:2}
    r_num = {int(k[1:]): int(v) for k, v in r.items()}
    kpis = model.compute_kpis(r_num)
    # weighted overall drop percent
    lambdas = {w: model.workflows[w]['lambda'] for w in model.workflows}
    total_lambda = sum(lambdas.values())
    total_dropped = sum((kpis[f"D_wf{w}/λ"]/100.0) * lambdas[w]
                        for w in model.workflows)
    overall_drop_pct = total_dropped / total_lambda * 100.0
    mean_node_util = sum(kpis[f"u_node{n}"] for n in range(model.N)) / model.N
    # save per-partition summary
    summary = {
        "partition": part,
        "overall_drop_pct": round(overall_drop_pct, 3),
        "mean_node_util_pct": round(mean_node_util, 3),
        "per_workflow_drop": {f"wf{w}": round(kpis[f"D_wf{w}/λ"],3) for w in model.workflows},
        "per_node_util": {f"n{n}": round(kpis[f"u_node{n}"],3) for n in range(model.N)},
    }
    out_rows.append(summary)
    out_path = out_dir / "metrics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")

# also write aggregate
agg_path = BASE_DIR / "aggregate_metrics.json"
agg_path.write_text(json.dumps(out_rows, indent=2))
print(f"Wrote {agg_path}")
