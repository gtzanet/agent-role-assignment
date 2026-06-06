# Experimental Plan

## Goal

Document the evaluation runs that support the thesis claim that TIG adapts to structure rather than always centralizing decisions.

## Config files to add

- `analytical_config_M4N3_A_isolated.yaml`
- `analytical_config_2S2N_B_striped.yaml`
- `analytical_config_M4N3_C_overloaded.yaml`
- `analytical_config_M4N3_B_striped_lambda15.yaml`
- `analytical_config_M4N3_B_striped_lambda25.yaml`
- `analytical_config_M4N3_B_striped_lambda35.yaml` (copy of the current `analytical_config.yaml`)

## Experiment sets

### 1. M4N3_A_isolated

Control condition.

Expected claim: TIG should recover the natural workflow-aligned structure where it exists and should not force unnecessary centralization on isolated workflows.

### 2. 2S2N_B_striped

Small verifiable demo.

Expected claim: node-aligned grouping should outperform workflow-aligned grouping, and TIG should find the global optimum on the exhaustive 4^4 assignment space.

### 3. M4N3_C_overloaded

C_max boundary test.

Expected claim: full centralization is infeasible, so TIG should produce the best feasible split with only a small gap from the unconstrained optimum.

### 4. Lambda sweep on M4N3_B_striped

Operating-regime sensitivity test.

Expected claim: the advantage of TIG should increase as load rises from light to heavy.

## Thesis updates

- Rewrite the evaluation setup section in `thesis-master/chapters/work3.tex` so each experiment family has its own subsection.
- Add a placeholder results table for each experiment family with `x` values where measured results will later be inserted.
- Replace the current generic claims with experiment-specific expected outcomes and conclusions.
