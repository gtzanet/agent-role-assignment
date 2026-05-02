import numpy as np
from typing import Dict

class ServiceSimulator:
    """
    Simulates a Service Deployment environment (Refined).
    2 Services (S1, S2), 2 Nodes (N1, N2).
    
    Inputs:
    - Replicas_S1, Replicas_S2: Number of replicas (normalized 0.0-1.0 for simplicity or int)
    - Node_S1, Node_S2: Placement (0 for N1, 1 for N2)
    
    Intermediaries 1:
    - Load_N1, Load_N2: Abstract load on node
    
    Intermediaries 2:
    - CPU_N1, CPU_N2: CPU utilization
    
    KPIs:
    - Lat_S1, Lat_S2: Average Latency
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_sample(self) -> Dict[str, float]:
        # Inputs: Replicas (0 to 10 scaled down to 0-1 for model, or just use raw)
        # Let's keep 0-1 float for consistency with previous "Scale", conceptualizing it as "Cluster Size %"
        replicas_s1 = self.rng.random() 
        replicas_s2 = self.rng.random()
        
        # Placement
        node_s1 = float(self.rng.choice([0, 1]))
        node_s2 = float(self.rng.choice([0, 1]))
        
        # Service Weights
        w_s1 = 2.0 # Heavy
        w_s2 = 1.5 # Moderate (Increased from 0.5 for visibility)
        
        # 1. Calculate Node Load
        load_n1 = 0.0
        load_n2 = 0.0
        
        # S1 Contribution
        if node_s1 < 0.5:
            load_n1 += replicas_s1 * w_s1
        else:
            load_n2 += replicas_s1 * w_s1
            
        # S2 Contribution
        if node_s2 < 0.5:
            load_n1 += replicas_s2 * w_s2
        else:
            load_n2 += replicas_s2 * w_s2
            
        # Add some noise to Load (background processes)
        load_n1 += self.rng.normal(0, 0.05)
        load_n2 += self.rng.normal(0, 0.05)
        # Ensure positive
        load_n1 = max(0.0, load_n1)
        load_n2 = max(0.0, load_n2)
        
        # 2. Calculate CPU from Load
        # Simple linear mapping: CPU = Base + 0.3 * Load
        # (Load can go up to ~2.5, so 0.3*2.5 = 0.75. + 0.1 base = 0.85 approx max)
        cpu_n1 = 0.1 + 0.3 * load_n1 + self.rng.normal(0, 0.01)
        cpu_n2 = 0.1 + 0.3 * load_n2 + self.rng.normal(0, 0.01)
        
        # Clamp CPU
        cpu_n1 = min(1.0, max(0.0, cpu_n1))
        cpu_n2 = min(1.0, max(0.0, cpu_n2))
        
        # 3. Calculate KPIs (Latency)
        # Latency = Base + Factor * CPU^2
        def calc_latency(cpu_load):
            return 20 + 100 * (cpu_load**2)

        if node_s1 < 0.5:
            lat_s1 = calc_latency(cpu_n1)
        else:
            lat_s1 = calc_latency(cpu_n2)
            
        if node_s2 < 0.5:
            lat_s2 = calc_latency(cpu_n1)
        else:
            lat_s2 = calc_latency(cpu_n2)
            
        lat_s1 += self.rng.normal(0, 0.5)
        lat_s2 += self.rng.normal(0, 0.5)
        
        return {
            "Replicas_S1": replicas_s1,
            "Replicas_S2": replicas_s2,
            "Node_S1": node_s1,
            "Node_S2": node_s2,
            "Load_N1": load_n1,
            "Load_N2": load_n2,
            "CPU_N1": cpu_n1,
            "CPU_N2": cpu_n2,
            "Lat_S1": lat_s1,
            "Lat_S2": lat_s2
        }

    def generate_batch(self, n_samples: int) -> Dict[str, np.ndarray]:
        keys = ["Replicas_S1", "Replicas_S2", "Node_S1", "Node_S2", 
                "Load_N1", "Load_N2", 
                "CPU_N1", "CPU_N2", 
                "Lat_S1", "Lat_S2"]
        data = {k: [] for k in keys}
        for _ in range(n_samples):
            sample = self.generate_sample()
            for k, v in sample.items():
                data[k].append(v)
        return {k: np.array(v) for k, v in data.items()}
