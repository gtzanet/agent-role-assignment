import pandas as pd
from simulation_services import ServiceSimulator

def main():
    print("=== Generating Services Dataset (Refined) ===")
    sim = ServiceSimulator()
    
    n_samples = 2000
    batch = sim.generate_batch(n_samples)
    
    df = pd.DataFrame(batch)
    
    output_file = "services_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved {n_samples} samples to {output_file}")
    
    # Verify Relations
    print("\n=== Verification: Load Logic ===")
    
    # Case 1: Replicas -> Load (Conditional)
    # Filter where S1 is on N1
    df_n1 = df[df["Node_S1"] == 0.0]
    
    print("\n[Condition: S1 on Node 1]")
    corr_s1_load1 = df_n1["Replicas_S1"].corr(df_n1["Load_N1"])
    corr_s1_load2 = df_n1["Replicas_S1"].corr(df_n1["Load_N2"])
    print(f"  Corr(Replicas_S1, Load_N1): {corr_s1_load1:.4f} (Expected High)")
    print(f"  Corr(Replicas_S1, Load_N2): {corr_s1_load2:.4f} (Expected Low)")

    # Case 2: Load -> CPU (Direct)
    # Should be high regardless of placement, as CPU is func of Load
    corr_load1_cpu1 = df["Load_N1"].corr(df["CPU_N1"])
    print("\n[Direct: Load -> CPU]")
    print(f"  Corr(Load_N1, CPU_N1): {corr_load1_cpu1:.4f} (Expected High)")

    # Case 3: CPU -> Latency (Conditional on Placement)
    print("\n[Condition: S1 on Node 1]")
    corr_cpu1_lat1 = df_n1["CPU_N1"].corr(df_n1["Lat_S1"])
    print(f"  Corr(CPU_N1, Lat_S1): {corr_cpu1_lat1:.4f} (Expected High)")

if __name__ == "__main__":
    main()
