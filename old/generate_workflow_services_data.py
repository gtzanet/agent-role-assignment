import pandas as pd
from workflow_services_simulator import WorkflowServiceSimulator


def main():
    print("=== Generating Workflow Services Dataset ===\n")
    
    # Initialize simulator with minimal parameters
    sim = WorkflowServiceSimulator(n_workflows=2, max_chain_length=2, n_nodes=2)
    
    # Show workflow structure
    print("Workflow Structure:")
    for wf_id, services in sim.workflows.items():
        service_info = " -> ".join([f"S{sid}(N{nid})" for sid, nid in services])
        print(f"  Workflow {wf_id}: {service_info}")
    
    print(f"\nGenerating 500 samples...")
    n_samples = 500
    all_data = sim.generate_batch(n_samples)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_file = "workflow_services_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved {len(df)} samples to {output_file}\n")
    
    # Print summary statistics
    print("=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Workflows: {sim.n_workflows}")
    print(f"Services: {max([int(col.split('_')[1]) for col in df.columns if col.startswith('Service_')]) + 1}")
    
    print("\n=== Sample Columns ===")
    print(sorted(df.columns))
    
    print("\n=== Sample Data (first 2 rows) ===")
    # Show just the replica and latency columns for brevity
    replica_cols = [col for col in df.columns if 'Replicas' in col]
    latency_cols = [col for col in df.columns if 'Latency' in col]
    print(df[replica_cols + latency_cols].head(2).to_string())


if __name__ == "__main__":
    main()
