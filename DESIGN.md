# Design: Causality-Driven Agent Task Decomposition

## Goal

In multi-agent orchestration systems for distributed compute infrastructures (edge/cloud/6G), a fundamental tension exists between two extremes:

- **Centralized control** — one agent handles all decisions. Optimal coordination, but action space grows exponentially with the number of services, and a single operator cannot always be assumed (multi-party systems, privacy boundaries).
- **Fully decentralized control** — one agent per decision. Tractable complexity, but agents act on local objectives without awareness of how their decisions jointly affect shared KPIs, leading to SLA violations.

The goal of this project is to determine, given a networked computing system, **which orchestration tasks should be assigned to which agents** — automatically and data-drivenly — so that:

1. Tasks that jointly influence the same KPIs stay together (minimizing coordination loss).
2. No single agent's joint action space exceeds a tractable complexity budget (minimizing SLA degradation from cognitive overload).

This is the **Agent Task Decomposition** problem, studied as part of thesis research on multi-agent orchestration.

---

## System Model

The system is modelled as a Causal Directed Acyclic Graph (DAG) `G = (V, E)` with three node types:

| Type | Symbol | Examples |
|------|--------|---------|
| **Input** (controllable) | `I` | `s0_avg_threads`, `s1_avg_threads` — decisions agents take |
| **Intermediary** (observable) | `X` | queue depth, per-service latency — internal system states |
| **KPI** (target outcomes) | `K` | `wf0_violation_rate`, `node0_cpu_usage_pct` |

Each input node `t_i` has a discrete action space `A_i`. A KPI weight vector `ω` reflects operator priorities.

The system seeks a task-to-agent assignment `map: I → {1…N}` and policies `π_a` for each agent `a`, maximising:

```
J = Σ_j ω_j · k̃_j
```

subject to each agent's joint decision space staying within `C_max`:

```
∏_{t_i ∈ G_a} |A_i| ≤ C_max   for all a
```

---

## Solution: Task Interaction Graph (TIG) + Graph Partitioning

### Step 1 — Causal Effect Estimation (Δ matrix)

For each `(task_i, kpi_j)` pair that are causally connected in the discovered DAG, estimate the **normalized interventional effect** using do-calculus:

```
Δ(i → j) = max_{x,x'} |E[kpi_j | do(task_i = x)] − E[kpi_j | do(task_i = x')]|
             ──────────────────────────────────────────────────────────────────
                                  kpi_j_max − kpi_j_min
```

This produces a delta matrix `Δ ∈ ℝ^{N×M}` (N tasks, M KPIs). Pairs with no causal path are skipped.

### Step 2 — Task Interaction Graph edge weights

For any two tasks `i` and `j`, compute a scalar `W_{i,j} ∈ [0, 1]` that measures the benefit of assigning them to the same agent:

**Pull force — Causal Footprint Alignment:**

```
c_i = [ω_1·Δ(i→k_1), …, ω_M·Δ(i→k_M)]        # KPI-weighted causal vector
F_pull(i,j) = (c_i · c_j) / (‖c_i‖ · ‖c_j‖)    # Cosine similarity ∈ [0,1]
```

High `F_pull` means the tasks influence the same KPIs in the same proportions — keeping them together avoids coordination loss.

**Push force — Complexity Penalty:**

```
C(i,j) = |A_i| × |A_j|
F_push(i,j) = σ(ρ · (C(i,j) − C_max))            # Logistic penalty ∈ (0,1)
```

High `F_push` means jointly assigning these tasks would exceed the agent's tractable complexity budget.

**Final edge weight:**

```
W_{i,j} = F_pull(i,j) · (1 − F_push(i,j))
```

This is symmetric, bounded in `[0,1]`, and naturally represents "how strongly should tasks i and j be co-assigned."

### Step 3 — Graph Partitioning

The weight matrix `W` is treated as the adjacency of an undirected Task Interaction Graph. Partitioning it into N agent groups is done with:

- **Greedy Modularity** (default): finds communities that maximise within-community edge weight.
- **Spectral Clustering**: uses eigendecomposition of the affinity matrix; suitable when `n_agents` is fixed.
- **Kernighan-Lin Bisection**: heuristic bisection optimising edge-cut; only for `n_agents=2`.

The result is a mapping `{agent_id → [task_list]}`, serialised as `agent_assignments.yaml`.

---

## Pipeline

The system runs as a five-stage sequential pipeline:

```
Stage 1   Workflow simulation     →  causal dataset (CSV)
Stage 2   Causal discovery        →  CausalGraph (DAG)
Stage 3   Causal inference + TIG  →  Δ matrix, W matrix
Stage 4   Graph partitioning      →  agent_assignments.yaml
Stage 5   RL evaluation           →  SLA violation / CPU usage metrics
```

### Stage 1 — Simulation & Data Collection (`generate_dataset.py`, `workflow_simulator/`)

Simulates a microservice application under time-varying Poisson workloads. Each service runs on a physical node with configurable CPU thread limits. The simulation collects per-service and per-workflow metrics at fixed eval intervals. One dataset row = one snapshot. The controllable inputs are thread counts (`s{i}_avg_threads`); outputs include end-to-end latency and workflow violation rates.

### Stage 2 — Causal Discovery (`causality/causal_discovery.py`)

Applies a causal discovery algorithm to the observational dataset:
- **NOTEARS** (default): continuous constrained optimisation for DAG structure.
- **LiNGAM**: exploits non-Gaussianity to identify causal direction.
- **Correlation**: threshold-based fallback.

Outputs a `CausalGraph` (`environment.py`), serialised to JSON for downstream stages.

### Stage 3 — Causal Inference & TIG (`causality/causal_inference.py`, `causality/task_interference_analyzer.py`)

Fits a discretised Bayesian network per connected component of the causal graph. For each reachable `(task, kpi)` pair, issues interventional queries (`do`-calculus) to compute `Δ`. Parallelises queries across tasks using `ThreadPoolExecutor`.

Then computes the full `N×N` weight matrix `W` using `compute_task_interaction_weight_matrix()` and saves it as `tig_W.csv`.

### Stage 4 — Partitioning (`causality/task_interference_analyzer.py: partition_tig()`)

Loads `tig_W.csv`, runs the selected partition algorithm, and writes:
- `partitions.json` — `{agent_id: [task_names]}`
- `agent_assignments.yaml` — sim-ready config fragment
- `partition_summary.json` — metadata

### Stage 5 — Evaluation (`run_experiment.py: evaluate_allocation()`)

Runs the simulator with RL agents (tile-coded Q-learning) trained under the selected task assignment, then evaluates against four baselines:

| Scenario | Description |
|----------|-------------|
| `selected` | TIG-derived partitioning |
| `one_agent_all_tasks` | Centralised: single agent for all services |
| `one_agent_per_task` | Fully decentralised: one agent per service |
| `one_agent_per_workflow` | One agent per workflow |
| `one_agent_per_node` | One agent per compute node |

Metrics collected: per-workflow SLA violation count and rate, per-node CPU usage. Results written as CSVs and JSON summaries per scenario.

---

## Key Files

| File | Role |
|------|------|
| `environment.py` | `CausalGraph`, `Node`, `NodeType` data model |
| `causality/causal_discovery.py` | NOTEARS / LiNGAM / correlation causal discovery |
| `causality/causal_inference.py` | Bayesian network fitting + interventional queries |
| `causality/task_interference_analyzer.py` | `compute_task_interaction_weight_matrix()`, `TaskInterferenceAnalyzer`, `partition_tig()` |
| `run_experiment.py` | Pipeline orchestration (stages 2–5) |
| `generate_dataset.py` | Stage 1 dataset generation |
| `workflow_simulator/` | Discrete-event microservice simulator + RL agents |
| `sim_utils.py` | Shared runtime builder, config loader |
| `workflow_causal_graph.yaml` | Node type and decision space metadata |
| `exp_config.yaml` | Experiment parameters (algorithm choices, thresholds, agent count) |

---

## Key Hyperparameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `C_max` | Max tractable joint decision space per agent | 100 |
| `rho` | Steepness of the logistic complexity penalty | 0.1 |
| `omega` | KPI criticality weights (uniform if omitted) | uniform |
| `delta_mode` | `interventional` (do-calculus) or `reachability` (binary) | interventional |
| `causal_algorithm` | `notears`, `lingam`, or `correlation` | correlation |
| `partition_algorithm` | `greedy_modularity`, `spectral`, or `kernighan_lin` | greedy_modularity |
| `n_agents` | Number of agent groups to create | 2 |

---

## Research Hypothesis

A task partition derived from causal footprint alignment and complexity constraints produces a multi-agent system that matches or exceeds centralized control on SLA satisfaction while remaining tractable at scale — and outperforms arbitrary (per-task, per-node, per-workflow) decompositions on coordination-sensitive workloads.

---

## Experimentation Plan

### Fixed system model

Each workflow has **2 services** (a chain s_root → s_leaf). Each service has one controllable input (`avg_threads`), action space size 4. With `C_max = 100`, at most 3 tasks fit in one agent (4³=64 ≤ 100; 4⁴=256 > 100). All runs use the same seed, RL hyperparameters, train episodes, and pipeline settings — one factor changes at a time.

---

### Design rationale: why mapping is the critical variable

Two KPI types exist in the system:
- **Workflow KPIs** (`wfX_violation_rate`): driven only by the 2 services of that workflow.
- **Node KPIs** (`nodeY_cpu_usage_pct`): driven by all services co-located on that node.

The service→node **mapping** determines whether tasks from *different* workflows share a node-level KPI. When they do, their causal footprints overlap and TIG should group them together — even though per-workflow and per-task baselines will not.

| Mapping type | Cross-workflow KPI coupling | Which baseline is near-optimal | What TIG must discover |
|---|---|---|---|
| Isolated | None (each workflow owns its node) | `per_workflow` ≈ `per_node` | Per-workflow structure |
| Striped | Strong (cross-workflow node sharing) | Neither — both fail | Node-aligned grouping |
| Overloaded | Maximal (few nodes, many services) | Centralized (if C_max permits) | Minimal viable grouping |

The **striped mapping is the critical condition** where all rigid baselines fail and TIG's causal analysis provides a genuine advantage.

---

### Dimension 1 — Scale (M workflows, N nodes)

| Label | M | N | Total tasks | C_max pressure | Purpose |
|---|---|---|---|---|---|
| S | 2 | 2 | 4 | Low | Minimal verifiable demo |
| M | 4 | 3 | 8 | Moderate | Primary configuration |
| L | 8 | 4 | 16 | High | Scalability |

M=4, N=3 is the **primary configuration**: C_max constraint is binding, the causal graph is interpretable, and the number of agents the algorithm produces is non-trivial.

---

### Dimension 2 — Mapping type (defined for M=4, N=3)

**Mapping A — Isolated** (control; per-workflow aligns with per-node)
```
wf0: s0, s1  →  node0
wf1: s2, s3  →  node1
wf2: s4, s5  →  node2
wf3: s6, s7  →  node2   ← wf2 and wf3 share node2
```
TIG should discover that wf2 and wf3 are coupled via node2 and group their tasks together. `per_workflow` fails on this pair; TIG should recover the correct structure.

**Mapping B — Striped** (adversarial; showcase configuration)
```
wf0: s0 → node0,  s1 → node1
wf1: s2 → node0,  s3 → node1
wf2: s4 → node1,  s5 → node2
wf3: s6 → node1,  s7 → node2
```
Node1 hosts services from all four workflows. `per_workflow` and `per_task` both impose the wrong grouping. TIG must group by node-coupling to minimise coordination loss.

**Mapping C — Overloaded** (C_max boundary test)
```
s0–s3  →  node0
s4–s7  →  node1
```
Centralised would be theoretically optimal but 4⁸ >> C_max. Tests whether TIG degrades gracefully rather than catastrophically when full centralisation is forbidden.

---

### Run matrix (priority order)

| Priority | M | N | Mapping | Purpose |
|---|---|---|---|---|
| 1 | 4 | 3 | B — Striped | Main result: TIG vs. per_workflow failure |
| 2 | 4 | 3 | A — Isolated | Control: TIG recovers natural structure |
| 3 | 2 | 2 | B — Striped | Minimal reproducible demo, verifiable by hand |
| 4 | 8 | 4 | B — Striped | Scalability |
| 5 | 4 | 3 | C — Overloaded | C_max boundary behaviour |

Start with Priority 1 and 3 in parallel. The small case lets you verify the causal graph is discovered correctly before trusting the larger-scale result.

---

### Metrics per run

For each (scale × mapping) configuration, compare the 5 scenarios (`selected`, `one_agent_all_tasks`, `one_agent_per_task`, `one_agent_per_workflow`, `one_agent_per_node`) on:

1. **Total SLA violations** — primary metric
2. **Per-workflow violation rate** — shows which workflows suffer
3. **Average node CPU usage** — resource efficiency
4. **Agent count and task compositions** — validates TIG discovered the intended structure

### Claims to support numerically

- TIG ≤ `one_agent_per_task` violations in all configurations
- TIG ≤ `one_agent_per_workflow` violations on striped mapping (main result)
- TIG ≤ `one_agent_per_node` violations because edge weights encode causal strength, not just topology
- TIG ≈ `one_agent_all_tasks` at small scale where C_max is non-binding
