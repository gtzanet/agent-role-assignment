## Game Matrix Equilibrium Plan

Implement `calc_equilibriums` in `evaluate_equilibriums.py` as a static game analyzer, not a trajectory simulator. The script should enumerate the joint action space of the partitioned services, build the payoff matrix for every joint profile, and then identify pure-strategy equilibria directly from that matrix.

### Utility Definition

For each agent and each joint profile:

- Primary utility `U_1`: the percentage of the agent’s services with `rho_s < 1`.
- Tie-break utility `U_2`: `sum(lam_eff[s] - C[s] for s in agent_services)`.

Rank profiles by maximizing `U_1`; if tied, minimize `U_2`; if still tied, use a deterministic fallback.

### Implementation Steps

1. Validate inputs and load the analytical model.
- Ensure `self.tig` and `self.assignment` are loaded.
- Instantiate `GPSModel(self.config.load())`.
- Read each agent’s service group and the corresponding replica action spaces.

2. Enumerate the joint-action game matrix.
- For each agent, enumerate the Cartesian product of that agent’s action space.
- Combine the per-agent choices into one full replica assignment.
- Evaluate each profile once with `model.compute_service_stats(r)`.
- Store each profile’s per-agent payoff vector and relevant state metrics.

3. Detect pure-strategy equilibria.
- For each joint profile, test every unilateral deviation by every agent.
- A profile is a pure Nash equilibrium if no agent can improve its payoff under the ranking above.
- Optionally record best-response regrets for diagnostic reporting.

4. Persist compact outputs.
- Save a table of all evaluated joint profiles and payoffs.
- Save a summary listing all equilibria and their replica assignments.
- Save lightweight diagnostics such as total profiles evaluated and number of equilibria found.
- Do not generate trajectory logs or convergence history.

5. Keep `main()` aligned with static analysis only.
- Print the scenario and partition.
- Print the number of joint profiles evaluated and the equilibria found.
- Avoid any output that implies iterative simulation.

### Relevant Files

- [evaluate_equilibriums.py](evaluate_equilibriums.py) — implement `calc_equilibriums` as static enumeration and equilibrium detection.
- [gps_model.py](gps_model.py) — reuse `compute_service_stats`, `compute_kpis`, and `score` for payoff/state calculations.
- [evaluate_partition_joint_actions.py](evaluate_partition_joint_actions.py) — reuse the exhaustive joint-action enumeration pattern.

### Verification

1. Run `python evaluate_equilibriums.py --config analytical_config.yaml`.
2. Confirm the script reports enumerated profiles and equilibria, not rounds or iterations.
3. Cross-check one equilibrium profile by manually verifying that no unilateral deviation improves the acting agent’s utility.
4. Confirm no trajectory-related files are created.

### Notes

- The scope is pure-strategy equilibrium detection.
- If mixed-strategy equilibria are needed later, that requires a different solver and output format.

---

## Experiments

Each experiment below defines a config YAML, the hypothesis it tests, and what to compare. All experiments are run via:

```bash
python evaluate_equilibriums.py --config <config_path>
```

Results land in `results/analytical/<name>/equilibriums/<algo>_<n_agents>/`.

### Metrics to compare across all experiments

| Metric | Definition | Good ↑/↓ |
|--------|-----------|----------|
| Num equilibria | Count of pure-strategy Nash equilibria | context-dependent |
| Worst-eq U1 (system) | Min across agents of `U1` in the worst equilibrium | ↑ higher = fewer SLA violations |
| Worst-eq U2 (system) | `U2` of the worst equilibrium (excess arrival) | ↓ lower = less overload |
| Avg-eq U1 (system) | Mean `U1` across all equilibria | ↑ higher = better average case |
| Avg-eq U2 (system) | Mean `U2` across all equilibria | ↓ lower = less average overload |
| Price of Anarchy (PoA) | `J(worst_equilibrium) / J(social_optimum)` | ↓ closer to 1.0 |

The script already compares every TIG-based partition against the four topology baselines (`all_in_one`, `all_separate`, `per_node`, `per_workflow`). Each experiment should therefore report the above metrics for *all* partition strategies.

---

### Experiment 1 — Minimal Contention Baseline (2S-2N isolated)

**Hypothesis**: When services do *not* share a node, partitioning has no effect because there is no resource contention. All partitions should yield identical equilibria.

**What it tests**: Validates that the framework correctly identifies a trivial case; serves as a sanity check.

**Config** (`GTA_experiments/exp1_2S2N_isolated.yaml`):

```yaml
name: exp1_2S2N_isolated

system:
  N: 2
  M: 2
  L: 1400

nodes:
  - id: 0
    f: 1400000
  - id: 1
    f: 1400000

delta: 28000

workflows:
  - id: 0
    lambda: 440.0
    services:
      - id: 0
        node: 0              # s0 alone on node0
        replicas_baseline: 1
        action_space: [1, 2, 3, 4, 5]
  - id: 1
    lambda: 440.0
    services:
      - id: 1
        node: 1              # s1 alone on node1
        replicas_baseline: 1
        action_space: [1, 2, 3, 4, 5]

tig:
  C_max: 100
  rho: 0.1
  kpi_weights:
    drop_rate:   0.5
    utilization: 0.5
  delta_method: "sii_marginalised"

partitioning:
  algorithm: "greedy_modularity"
  n_agents: 2
```

**Expected outcome**: TIG W matrix is near-zero off-diagonal. All partitions produce the same equilibria (no benefit to grouping). `all_separate` and `per_workflow` are equivalent and optimal.

---

### Experiment 2 — Shared-Node Contention (2S-1N, existing)

**Hypothesis**: When two services share a single node, adding replicas to one steals capacity from the other. Partition matters: `all_separate` creates a prisoner's dilemma where each agent selfishly over-allocates, leading to a bad equilibrium. `all_in_one` (centralized) avoids this.

**What it tests**: Core contention dynamics; the fundamental tragedy-of-the-commons scenario.

**Config**: Use the existing `results/analytical/queueing_game_2S1N/config.yaml` (2 services, 1 node, λ=440 each).

**Key comparison**: 
- `all_in_one` (1 agent): should find the social optimum as the only equilibrium.
- `all_separate` (2 agents): should exhibit a worse equilibrium (both over-allocate replicas, U1 drops).
- TIG-based partition: with `n_agents=1`, TIG should recognize strong interaction and keep them together.

**Expected outcome**: `all_separate` worst-eq has U1 < `all_in_one` worst-eq U1. TIG groups them into one agent (equivalent to `all_in_one`).

---

### Experiment 3 — Striped Topology (4WF-3N, existing core case)

**Hypothesis**: When workflows are "striped" across nodes (each workflow's stages land on different nodes, and a hub node is shared by multiple workflows), `per_workflow` and `per_node` both fail — only a TIG-aware partition captures the cross-cutting contention.

**What it tests**: The primary thesis scenario; demonstrates that causal/interaction-aware partitioning outperforms topology baselines.

**Config**: Use the existing `results/analytical/M4N3_B_striped/config.yaml` (8 services, 3 nodes, 4 workflows, node1 is the hub hosting services from all workflows).

**Key comparison**:
- `per_workflow` (4 agents): isolates workflows but ignores node1 contention.
- `per_node` (3 agents): captures co-location but ignores workflow chains.
- TIG `spectral` (3 agents): should group services that share node1 resources, outperforming both baselines.

**Expected outcome**: TIG partition worst-eq U1 ≥ baselines. If not, investigate whether `C_max`/`rho` TIG parameters need tuning.

---

### Experiment 4 — Asymmetric Load Stress Test (2S-1N, skewed λ)

**Hypothesis**: When one workflow has much higher arrival rate than the other, the high-λ service dominates the shared node. Partitioning into separate agents causes the low-λ agent to be starved because the high-λ agent takes most replicas selfishly.

**What it tests**: Whether TIG correctly identifies asymmetric contention and keeps the services together under one agent.

**Config** (`GTA_experiments/exp4_2S1N_asymmetric.yaml`):

```yaml
name: exp4_2S1N_asymmetric

system:
  N: 1
  M: 2
  L: 1400

nodes:
  - id: 0
    f: 1400000

delta: 28000

workflows:
  - id: 0
    lambda: 700.0            # high-load workflow (~73% of total)
    services:
      - id: 0
        node: 0
        replicas_baseline: 1
        action_space: [1, 2, 3, 4, 5]
  - id: 1
    lambda: 250.0            # low-load workflow (~27% of total)
    services:
      - id: 1
        node: 0
        replicas_baseline: 1
        action_space: [1, 2, 3, 4, 5]

tig:
  C_max: 100
  rho: 0.1
  kpi_weights:
    drop_rate:   0.5
    utilization: 0.5
  delta_method: "sii_marginalised"

partitioning:
  algorithm: "greedy_modularity"
  n_agents: 1
```

**Expected outcome**: `all_separate` creates a very bad worst equilibrium (high-λ agent hogs replicas). TIG detects strong asymmetric interaction and assigns both to one agent, recovering the social optimum.

---

### Experiment 5 — Scaling: Larger Service Count (6S-2N, 3 workflows)

**Hypothesis**: As the system grows, the joint action space explodes. TIG-based partitioning should yield fewer equilibria with better worst-case guarantees than topology baselines, because it groups interacting services while isolating independent ones.

**What it tests**: Scalability of equilibrium quality; whether TIG works with multi-stage workflows.

**Config** (`GTA_experiments/exp5_6S2N_chain.yaml`):

```yaml
name: exp5_6S2N_chain

system:
  N: 2
  M: 3
  L: 10

nodes:
  - id: 0
    f: 1000
  - id: 1
    f: 1000

delta: 5

workflows:
  - id: 0
    lambda: 30.0
    services:
      - id: 0
        node: 0
        replicas_baseline: 2
        action_space: [1, 2, 3]
      - id: 1
        node: 1
        replicas_baseline: 2
        action_space: [1, 2, 3]
  - id: 1
    lambda: 30.0
    services:
      - id: 2
        node: 0              # shares node0 with s0
        replicas_baseline: 2
        action_space: [1, 2, 3]
      - id: 3
        node: 1              # shares node1 with s1
        replicas_baseline: 2
        action_space: [1, 2, 3]
  - id: 2
    lambda: 30.0
    services:
      - id: 4
        node: 0              # shares node0 with s0, s2
        replicas_baseline: 2
        action_space: [1, 2, 3]
      - id: 5
        node: 1              # shares node1 with s1, s3
        replicas_baseline: 2
        action_space: [1, 2, 3]

tig:
  C_max: 50
  rho: 0.1
  kpi_weights:
    drop_rate:   0.5
    utilization: 0.5
  delta_method: "sii_marginalised"

partitioning:
  algorithm: "spectral"
  n_agents: 2
```

**Expected outcome**: `per_workflow` (3 agents) splits co-located services; `per_node` (2 agents) groups all co-located services naively. TIG `spectral` (2 agents) should match or beat `per_node` by considering interaction strength rather than just topology. Joint action space: 3^6 = 729 profiles, which is tractable for exhaustive enumeration.

---

### Experiment 6 — TIG Sensitivity: Varying `n_agents` on the Striped Topology

**Hypothesis**: The number of agents is a key parameter. Fewer agents → lower PoA but larger per-agent action spaces. More agents → smaller action spaces but higher PoA (more coordination failures). There should be an optimal agent count that TIG partitioning can exploit.

**What it tests**: Sensitivity of equilibrium quality to agent count; whether TIG partitioning degrades gracefully.

**Config**: Reuse the M4N3_B_striped config but sweep `n_agents`:

| Run | `n_agents` | `algorithm` | What it reveals |
|-----|-----------|-------------|-----------------|
| 6a  | 1 | `all_in_one` | Social optimum (centralized upper bound) |
| 6b  | 2 | `spectral` | TIG at 2 agents — best trade-off? |
| 6c  | 3 | `spectral` | TIG at 3 agents (matching `per_node`) |
| 6d  | 4 | `spectral` | TIG at 4 agents (matching `per_workflow`) |
| 6e  | 8 | `all_separate` | Fully decentralized (lower bound) |

For runs 6b–6d, create configs or pass overrides to the partitioning section:

```yaml
# Example for run 6b
partitioning:
  algorithm: "spectral"
  n_agents: 2
```

**Expected outcome**: PoA increases monotonically with `n_agents`. TIG (`spectral`) at each agent count should yield PoA ≤ the corresponding topology baseline with the same agent count. The sweet spot (best worst-eq U1 per unit of action-space reduction) should be at 2 or 3 agents.

---

### Running all experiments

```bash
# Exp 1 — Isolated services (sanity check)
python evaluate_equilibriums.py --config GTA_experiments/exp1_2S2N_isolated.yaml

# Exp 2 — Shared-node contention (existing config)
python evaluate_equilibriums.py --config results/analytical/queueing_game_2S1N/config.yaml

# Exp 3 — Striped topology (existing config)
python evaluate_equilibriums.py --config results/analytical/M4N3_B_striped/config.yaml

# Exp 4 — Asymmetric load
python evaluate_equilibriums.py --config GTA_experiments/exp4_2S1N_asymmetric.yaml

# Exp 5 — 6-service scaling
python evaluate_equilibriums.py --config GTA_experiments/exp5_6S2N_chain.yaml

# Exp 6 — Agent-count sweep (modify partitioning.n_agents for each run)
for n in 2 3 4; do
  # Create a temp config or modify in-place
  cp results/analytical/M4N3_B_striped/config.yaml /tmp/exp6_${n}.yaml
  sed -i '' "s/n_agents: .*/n_agents: $n/" /tmp/exp6_${n}.yaml
  python evaluate_equilibriums.py --config /tmp/exp6_${n}.yaml
done
```

### Summary Table

| Exp | Name | Services | Nodes | Workflows | Key Variable | Primary Question |
|-----|------|----------|-------|-----------|-------------|-----------------|
| 1 | Isolated baseline | 2 | 2 | 2 | No contention | Does the framework correctly show no partition benefit? |
| 2 | Shared-node (2S1N) | 2 | 1 | 2 | Node sharing | Does TIG detect the prisoner's dilemma? |
| 3 | Striped (M4N3) | 8 | 3 | 4 | Hub node | Does TIG outperform per-node and per-workflow? |
| 4 | Asymmetric load | 2 | 1 | 2 | Skewed λ | Does TIG handle load asymmetry? |
| 5 | 6S-2N chains | 6 | 2 | 3 | Scale + contention | Does TIG scale to multi-stage systems? |
| 6 | Agent-count sweep | 8 | 3 | 4 | `n_agents` | What's the optimal agent count for TIG? |
