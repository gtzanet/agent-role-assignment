# Agent Task Distribution System

## Project Overview

This repository implements a **multi-stage agent task distribution system** for optimizing resource allocation in microservice-based distributed systems. The system uses causal inference and graph analysis to automatically partition decision tasks among multiple agents, minimizing coordination overhead while maintaining KPI (Key Performance Indicator) compliance.

**Core Goal**: Determine which distribution control decisions should be assigned to which agents, using causal relationships and task interaction weights to minimize conflicts and improve scalability.

---

## Pipeline Architecture

The system operates through five sequential stages:

### Stage 1: Workflow Simulation & Data Collection
**Component**: `workflow_simulator/` module with `sim_utils.py`

**Purpose**: Generate observational data from a distributed system under realistic conditions.

**Process**:
1. Simulates a microservice-based workflow system with:
   - N services interconnected by a task graph
   - Time-varying Poisson arrivals for workflows
   - Per-service resource constraints (CPU threads, memory)
   - Discrete time-step events (service execution, queue transitions)

2. Collects metrics at configurable intervals:
   - Per-service: latency, throughput, queue size, thread count, arrival/departure rates
   - Per-workflow: end-to-end latency, SLA violation rates
   - Per-node: CPU utilization

3. Aggregates into **causal dataset** (one row = one simulation snapshot):
   - Input features: `s{i}_avg_threads` (controllable decisions per service)
   - Intermediary features: service performance metrics and queue states
   - KPI targets: `wf{j}_avg_e2e_latency`, `wf{j}_violation_rate`, `node{k}_cpu_usage_pct`

**Output**: `causal_dataset.csv` with hundreds/thousands of observational samples

**Configuration**: `config.yaml` defines topology, infrastructure, simulation parameters

---

### Stage 2: Causal Discovery
**Component**: `causality/causal_discovery.py`

**Purpose**: Infer the causal structure linking controllable decisions to KPIs.

**Process**:
1. Accepts raw observational data from Stage 1
2. Applies causal discovery algorithms:
   - **NoTEARS**: Continuous optimization to find acyclic causal DAG
   - **Correlation**: Simple correlation-based fallback
   - **LiNGAM**: Linear non-Gaussian acyclic models for stronger assumptions

3. Learns which input variables (thread counts) causally influence which KPIs
4. Identifies intermediary variables and their dependencies

**Output**: `CausalGraph` object encoded in `workflow_causal_graph.yaml`:
```yaml
causal_graph:
  node_types:
    s0_avg_threads: input           # Decisions
    s1_avg_latency: intermediary    # System states
    wf0_avg_e2e_latency: kpi        # Outcomes
  decision_space_sizes:
    s0_avg_threads: 4  # Range [0, 3] for each decision
```

---

### Stage 3: Causal Inference & Interventional Analysis
**Component**: `causality/causal_inference.py` with `causality/task_interference_analyzer.py`

**Purpose**: Quantify the causal effect of each decision on each KPI (interventional distribution estimation).

**Process**:
1. **Discretization**: Bin continuous data into discrete states for Bayesian network estimation
2. **Component decomposition**: Identify connected causal components
3. **Bayesian network estimation**: Learn conditional dependencies for each component
4. **Interventional querying**: For each decision `x_i` and KPI `y_j`, estimate $\mathbb{E}[Y_j | do(X_i = a)]$ across all values of $a$

5. **Delta matrix formation**: $\Delta \in \mathbb{R}^{N \times M}$ where:
   - $N$ = number of input decisions
   - $M$ = number of KPIs
   - $\Delta[i,j]$ = normalized causal effect magnitude of decision $i$ on KPI $j$

6. **Task Interaction Weight Matrix (W)**: Computed using `compute_task_interaction_weight_matrix()`:
   - Combines delta matrix with KPI criticality weights ($\omega$)
   - Computes cosine similarity ("pull") between causal effect vectors
   - Penalizes high decision space complexity ($A[i] \times A[j]$ per task pair)
   - Result: $W \in \mathbb{R}^{N \times N}$, symmetric, zero diagonal
   - Interpretation: $W[i,j]$ = interaction strength between decisions $i$ and $j$

**Output**: `tig_W.csv` (interaction matrix), `tig_delta.csv` (causal effects), `tig_summary.json` (metadata)

---

### Stage 4: Agent Partitioning & Task Assignment
**Component**: Spectral clustering logic in `run_workflow_experiment.py`

**Purpose**: Partition decisions into agent domains to minimize cross-agent coordination.

**Process**:
1. **Graph construction**: Treat task interaction matrix $W$ as weighted graph adjacency matrix
2. **Spectral clustering**: Apply `sklearn.cluster.SpectralClustering` with:
   - Number of clusters = number of agents
   - Affinity matrix derived from $W$
   - Objective: maximize within-cluster similarity, minimize between-cluster links

3. **Assignment**: Each cluster becomes one agent's domain
   - Agent 1 controls decisions {s0, s1}
   - Agent 2 controls decisions {s2, s3}
   - Etc.

4. **Human-readable output**:
   - `agent_assignments.yaml`: Maps agents to their services/decisions
   - `partition_assignments.csv`: Explicit assignment per decision
   - `partition_summary.json`: Statistics (cluster sizes, inter-cluster edges)

**Output**: `agent_assignments.yaml`, `partition_assignments.csv`, `partition_summary.json`

---

### Stage 5: Evaluation & Visualization
**Component**: `eval_experiment.ipynb` / `plot_results.ipynb`

**Purpose**: Analyze and visualize results across experiments and compare architectures.

**Process**:
1. **Load experiment results** from timestamped directories in `experiments/`
2. **Parse multi-scenario outputs**: Each directory may contain results for different configurations
3. **Compute metrics**:
   - Per-scenario: mean violation rates, latency, CPU usage
   - Per-agent partition: communication overhead (inter-cluster edges)
   - Scalability: how metrics degrade with agent count

4. **Visualization**:
   - Causal graph structure (nodes colored by type)
   - Interaction weight heatmaps
   - Agent partition boundaries overlay on graph
   - Metric comparison plots across scenarios

5. **Statistical comparison**: Baseline vs. multi-agent architectures

**Output**: Plots, summary tables, comparison matrices for research publication

---

## Key Data Structures

### CausalGraph (`environment.py`)
```python
class CausalGraph:
    nodes: Dict[str, Node]        # name -> Node(name, type, decision_space_size)
    graph: nx.DiGraph()           # Directed acyclic graph of causal relationships
```

**Node Types**:
- `INPUT`: Controllable decisions (thread counts per service)
- `INTERMEDIARY`: System state observations (latencies, queue sizes)
- `KPI`: Outcome metrics (violation rates, CPU usage)

### Experiment Output Structure
```
experiments/
├── 20260410_144452/              # Timestamp directory
│   ├── causal_dataset.csv         # Stage 1 output
│   ├── causal_graph.json          # Stage 2 causal structure
│   ├── tig_W.csv                  # Stage 3 interaction matrix
│   ├── tig_delta.csv              # Stage 3 causal effects
│   ├── agent_assignments.yaml     # Stage 4 partition assignment
│   ├── partition_summary.json     # Stage 4 statistics
│   └── causal_graph_summary.json  # Human-readable graph info
```

---

## Entry Points for Agents

1. **Full pipeline execution**:
   ```bash
   python run_workflow_experiment.py --num-agents 2 --num-samples 1000
   ```
   Runs all stages (1-4) sequentially, saves results to `experiments/TIMESTAMP/`

2. **Individual stages**:
   - Stage 1: `workflow_simulator/build_causal_dataset.py`
   - Stage 2: `causality/causal_discovery.py` + `infer_causal_graph_*()`
   - Stage 3: `causality/task_interference_analyzer.py` + `compute_task_interaction_weight_matrix()`
   - Stage 4: Use `SpectralClustering` on $W$ matrix
   - Stage 5: `eval_experiment.ipynb`

3. **Configuration**:
   - `workflow_simulator/config.yaml`: Simulation parameters (services, workflows, topology)
   - `workflow_causal_graph.yaml`: Pre-cached causal graph structure

---

## Key Parameters

| Parameter | Location | Meaning |
|-----------|----------|---------|
| `n_services` | config.yaml | Number of microservices in the application |
| `n_nodes` | config.yaml | Number of physical nodes in the cluster |
| `cpu_max` | config.yaml | Max CPU threads per node (per-service scaling range) |
| `iterations` | config.yaml | Simulation timesteps per episode |
| `bins` | run_workflow_experiment.py | Discretization granularity for Bayesian nets (default: 5) |
| `n_agents` | run_workflow_experiment.py | Number of agent partitions to create |
| `n_samples` | run_workflow_experiment.py | Number of simulation samples to collect |

---

## Research Context

This system is part of **thesis research** on decentralized multi-agent task distribution. The hypothesis: using causal analysis to inform agent partition assignment yields scheduling decisions with lower SLA violations and better resource utilization than baseline (centralized or random) approaches.

**Files**:
- `66e55b6e7d6816b1b1f32663/thesis-master/`: Full thesis document
- `old/`: Archived experiments and earlier implementations
