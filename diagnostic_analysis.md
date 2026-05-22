# Why Service A (1 replica) Doesn't Fail at λ=400: Model Mismatch Analysis

## The Paradox

**Full-Load Model Prediction**: Service A should fail (ρ_A = 1.32 > 1)
**GPS-PS Model Prediction**: System is stable (ρ_GPS = 0.88 < 1)
**Simulation Result**: System is stable ✓ (matches GPS-PS)

---

## Root Cause: Two Different Models

### GPS-PS Model (Simulator Implementation)
- **Assumption**: Speed adapts dynamically based on actual concurrency
  ```
  speed = FREQ / (n_running × (1 + OVERHEAD))
  ```
  where `n_running` is the number of tasks actually running (≤ R_total)

- **Why it's optimistic**: Not all threads run simultaneously
  - When only 2 tasks run: speed = 1.4M / (2 × 1.1) = 636 kop/s
  - When 3 tasks run: speed = 1.4M / (3 × 1.1) = 424 kop/s
  - Average speed > minimum speed

- **Current scenario**: 
  - Total utilization: ρ_GPS = (400 + 400) × 0.0011 = 0.88 < 1.0 ✓
  - System is stable

### Full-Load Model (Design Document)
- **Assumption**: All R_total = 3 threads always run simultaneously
  ```
  speed = FREQ / (R_total × (1 + OVERHEAD)) = constant
  ```

- **Why it's pessimistic**: Assumes worst case where all threads are competing

- **Current scenario**:
  - Service A: ρ_A = 400 / 303 = 1.32 > 1.0 ✗
  - Service A would be unstable IF all threads were always active

---

## Why the Simulation Doesn't Fail

The simulator uses **GPS-PS logic** with dynamic speed adjustment:

1. Tasks arrive according to Poisson processes
2. At any time, only `n_running ≤ R_total` tasks are executing
3. Speed automatically increases when fewer tasks compete
4. This adaptive mechanism keeps the system stable even though:
   - Service A has only 1 replica
   - Individual service capacities seem insufficient

---

## To Force Actual Failure

### Option 1: Increase Total Load Beyond GPS-PS Threshold

Current: λ_total = 800 req/s, ρ_GPS = 0.88

**Failure point**: ρ_GPS → 1.0 requires λ_total ≥ 900 req/s

**Test case**: λ_A = 500, λ_B = 400 (total 900)
- GPS-PS: ρ = 0.99 (approaching instability)
- Expected: Queues grow, latencies spike, system approaches collapse

### Option 2: Increase Overhead (More Context-Switching Cost)

Current: OVERHEAD = 0.1 (10% per thread)

**Test case**: OVERHEAD = 0.2 (20%)
- D increases from 1.1 ms to 1.68 ms
- Same load 800 req/s → ρ_GPS = 1.34 (unstable immediately)

### Option 3: Reduce Thread Pool

Current: (r_A=1, r_B=2) on shared node

**Test case**: Single-threaded system (R_total=1)
- Speed at full utilization: FREQ / (1 × 1.1) = 1.27M op/s
- System collapses for any arrival rate > 909 req/s

---

## Summary Table

| Scenario | λ_A | λ_B | λ_total | ρ_GPS | Status | Why |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Current (1,2) config | 400 | 400 | 800 | 0.88 | ✓ Stable | GPS-PS < 1.0 |
| High load symmetric | 450 | 450 | 900 | 0.99 | Edge | GPS-PS ≈ 1.0 |
| Very high on A | 500 | 400 | 900 | 0.99 | Edge | GPS-PS ≈ 1.0 |
| Extreme on A | 550 | 350 | 900 | 0.99 | Edge | GPS-PS ≈ 1.0 |
| **Higher overhead** | 400 | 400 | 800 | **1.34** | **✗ Fails** | Higher per-task demand |

---

## Conclusion

Service A doesn't fail because:
1. The simulator implements GPS-PS with **adaptive speed**
2. Not all threads run simultaneously
3. System utilization (0.88) is still below threshold (1.0)

To make Service A fail, you need to **exceed the GPS-PS stability threshold**, which requires either:
- Higher arrival rates (~900 req/s total)
- Higher overhead (context-switching cost)
- Fewer total threads

The full-load model is a **safety analysis** that ensures stability even in worst-case (all threads active). The GPS-PS model is the **actual operating point**.
