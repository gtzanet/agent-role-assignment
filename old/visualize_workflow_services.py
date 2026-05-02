import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_workflow_services_dataset():
    """Visualize the workflow services dataset."""
    
    # Load dataset
    df = pd.read_csv('workflow_services_dataset.csv')
    
    print(f"Loading {len(df)} records from workflow_services_dataset.csv...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Arrival Rate vs Service Rate (colored by Workflow)
    ax1 = plt.subplot(2, 3, 1)
    for wf_id in sorted(df['Workflow_ID'].unique()):
        df_wf = df[df['Workflow_ID'] == wf_id].drop_duplicates(subset=['Service_ID']).head(1)
        ax1.scatter(df_wf['Arrival_Rate'], df_wf['Service_Rate'], 
                   s=100, alpha=0.7, label=f'WF{wf_id}')
    
    # Overall scatter with transparency
    df_unique = df.drop_duplicates(subset=['Service_ID'])
    ax1.scatter(df_unique['Arrival_Rate'], df_unique['Service_Rate'], 
               s=50, alpha=0.3, c='gray', label='All services')
    ax1.set_xlabel('Arrival Rate (req/s)', fontsize=10)
    ax1.set_ylabel('Service Rate (req/s)', fontsize=10)
    ax1.set_title('Arrival vs Service Rate by Workflow', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Replicas distribution by Service
    ax2 = plt.subplot(2, 3, 2)
    replica_stats = df.groupby('Service_ID')['Replicas'].mean().sort_index()
    ax2.bar(replica_stats.index, replica_stats.values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Service ID', fontsize=10)
    ax2.set_ylabel('Average Replicas', fontsize=10)
    ax2.set_title('Average Replicas per Service', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Node distribution
    ax3 = plt.subplot(2, 3, 3)
    node_counts = df.drop_duplicates(subset=['Service_ID']).groupby('Node').size()
    colors = plt.cm.Set3(np.linspace(0, 1, len(node_counts)))
    ax3.pie(node_counts.values, labels=[f'Node {i}' for i in node_counts.index], 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Service Distribution across Nodes', fontsize=12, fontweight='bold')
    
    # 4. Latency distribution by Workflow
    ax4 = plt.subplot(2, 3, 4)
    latency_by_wf = [df[df['Workflow_ID'] == wf_id]['Latency_ms'].values 
                     for wf_id in sorted(df['Workflow_ID'].unique())]
    bp = ax4.boxplot(latency_by_wf, labels=[f'WF{i}' for i in sorted(df['Workflow_ID'].unique())],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(latency_by_wf)))):
        patch.set_facecolor(color)
    ax4.set_xlabel('Workflow', fontsize=10)
    ax4.set_ylabel('Service Latency (ms)', fontsize=10)
    ax4.set_title('Service Latency Distribution by Workflow', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. E2E Latency by Workflow
    ax5 = plt.subplot(2, 3, 5)
    e2e_stats = df.groupby('Workflow_ID')['E2E_Latency_ms'].agg(['mean', 'std']).reset_index()
    ax5.bar(e2e_stats['Workflow_ID'], e2e_stats['mean'], 
           yerr=e2e_stats['std'], capsize=5, color='coral', alpha=0.7, error_kw={'elinewidth': 2})
    ax5.set_xlabel('Workflow ID', fontsize=10)
    ax5.set_ylabel('E2E Latency (ms)', fontsize=10)
    ax5.set_title('End-to-End Latency by Workflow', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Utilization (Arrival / Service Rate) by Service
    ax6 = plt.subplot(2, 3, 6)
    df['Utilization'] = df['Arrival_Rate'] / df['Service_Rate']
    util_by_svc = df.groupby('Service_ID')['Utilization'].mean().sort_index()
    colors_util = ['red' if u > 0.7 else 'orange' if u > 0.5 else 'green' for u in util_by_svc.values]
    ax6.bar(util_by_svc.index, util_by_svc.values, color=colors_util, alpha=0.7)
    ax6.axhline(y=0.7, color='red', linestyle='--', label='High utilization (0.7)', alpha=0.5)
    ax6.axhline(y=0.5, color='orange', linestyle='--', label='Medium utilization (0.5)', alpha=0.5)
    ax6.set_xlabel('Service ID', fontsize=10)
    ax6.set_ylabel('Average Utilization', fontsize=10)
    ax6.set_title('Service Utilization (Arrival/Service Rate)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('workflow_services_visualization.png', dpi=300, bbox_inches='tight')
    print("[INFO] Saved visualization to workflow_services_visualization.png")
    plt.show()
    
    # Print detailed statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total records: {len(df)}")
    print(f"Unique workflows: {df['Workflow_ID'].nunique()}")
    print(f"Unique services: {df['Service_ID'].nunique()}")
    print(f"Nodes: {sorted(df['Node'].unique())}")
    
    print("\n=== Latency Statistics (ms) ===")
    print(f"Service Latency - Mean: {df['Latency_ms'].mean():.2f}, Std: {df['Latency_ms'].std():.2f}, "
          f"Min: {df['Latency_ms'].min():.2f}, Max: {df['Latency_ms'].max():.2f}")
    print(f"E2E Latency - Mean: {df['E2E_Latency_ms'].mean():.2f}, Std: {df['E2E_Latency_ms'].std():.2f}, "
          f"Min: {df['E2E_Latency_ms'].min():.2f}, Max: {df['E2E_Latency_ms'].max():.2f}")
    
    print("\n=== Utilization Statistics ===")
    print(f"Mean utilization: {df['Utilization'].mean():.3f}")
    print(f"Min utilization: {df['Utilization'].min():.3f}")
    print(f"Max utilization: {df['Utilization'].max():.3f}")
    print(f"Services with high utilization (>0.7): {(df['Utilization'] > 0.7).sum() // df['Service_ID'].nunique()}")
    
    print("\n=== Workflow Details ===")
    for wf_id in sorted(df['Workflow_ID'].unique()):
        df_wf = df[df['Workflow_ID'] == wf_id].drop_duplicates(subset=['Service_ID'])
        e2e = df_wf['E2E_Latency_ms'].iloc[0]
        services = sorted(df_wf['Service_ID'].unique())
        print(f"Workflow {wf_id}: {len(services)} services, E2E Latency mean: {e2e:.2f}ms")


if __name__ == "__main__":
    visualize_workflow_services_dataset()
