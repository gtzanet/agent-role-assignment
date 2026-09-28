#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# GTA Experiments — run all equilibrium evaluations from GTA_PLAN.md
# Usage:  bash GTA_experiments/run_all.sh
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate the project venv
source .venv/bin/activate

EVAL_SCRIPT="evaluate_equilibriums.py"
RESULTS_DIR="results/analytical"
LOG_DIR="GTA_experiments/logs"
mkdir -p "$LOG_DIR"

# Timestamp for this run
TS=$(date +%Y%m%d_%H%M%S)

run_experiment() {
    local label="$1"
    local config="$2"
    local logfile="$LOG_DIR/${label}_${TS}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ ${label}"
    echo "    config : ${config}"
    echo "    log    : ${logfile}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if python "$EVAL_SCRIPT" --config "$config" 2>&1 | tee "$logfile"; then
        echo "  ✔ ${label} completed"
    else
        echo "  ✘ ${label} FAILED (see ${logfile})"
    fi
    echo ""
}

# ── Experiment 1 — Minimal Contention Baseline (2S-2N isolated) ──
run_experiment "exp1_isolated" \
    "GTA_experiments/exp1_2S2N_isolated.yaml"

# ── Experiment 2 — Shared-Node Contention (2S-1N, existing) ──────
run_experiment "exp2_shared_node" \
    "$RESULTS_DIR/queueing_game_2S1N/config.yaml"

# ── Experiment 3 — Striped Topology (4WF-3N, existing) ───────────
run_experiment "exp3_striped" \
    "$RESULTS_DIR/M4N3_B_striped/config.yaml"

# ── Experiment 4 — Asymmetric Load Stress Test (2S-1N, skewed λ) ─
run_experiment "exp4_asymmetric" \
    "GTA_experiments/exp4_2S1N_asymmetric.yaml"

# ── Experiment 5 — Scaling: 6S-2N, 3 workflows ──────────────────
run_experiment "exp5_scaling" \
    "GTA_experiments/exp5_6S2N_chain.yaml"

# ── Experiment 6 — Agent-count sweep on Striped Topology ─────────
run_experiment "exp6b_striped_2agents" \
    "GTA_experiments/exp6b_M4N3_striped_2agents.yaml"

run_experiment "exp6c_striped_3agents" \
    "GTA_experiments/exp6c_M4N3_striped_3agents.yaml"

run_experiment "exp6d_striped_4agents" \
    "GTA_experiments/exp6d_M4N3_striped_4agents.yaml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All experiments finished.  Logs in: $LOG_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
