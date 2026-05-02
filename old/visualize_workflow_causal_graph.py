import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from causalnex.structure.notears import from_pandas

def infer_and_visualize_causal_graph(dataset_path: str, threshold: float = 0.05):
    """
    Infers and visualizes the causal graph from the workflow services dataset.
    """
    print(f"[INFO] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    labels = df.columns.tolist()
    
    print(f"[INFO] Variables: {len(labels)} columns")
    
    # Define Tier Constraints for Causal Structure
    # Tier 0: Inputs (Replicas)
    # Tier 1: Request flow rates (Arrival Rate, Service Rate)
    # Tier 2: Node CPU usage
    # Tier 3: Performance outcomes (Latency)
    # Tier 4: E2E outcomes (E2E Latency)
    
    tiers = {}
    for label in labels:
        if "Replicas" in label:
            tiers[label] = 0  # Input - controlled variables
        elif "Arrival_Rate" in label or "Service_Rate" in label:
            tiers[label] = 1  # Request flow rates
        elif "CPU" in label:
            tiers[label] = 2  # Node CPU usage
        elif "Latency_ms" in label and "E2E" not in label:
            tiers[label] = 3  # Service latency
        elif "E2E_Latency" in label:
            tiers[label] = 4  # End-to-end latency
        else:
            tiers[label] = 5  # Unknown
    
    print("\n[INFO] Variable Tiers:")
    tier_summary = {}
    for label, tier in sorted(tiers.items(), key=lambda x: x[1]):
        if tier not in tier_summary:
            tier_summary[tier] = 0
        tier_summary[tier] += 1
    for tier in sorted(tier_summary.keys()):
        print(f"  Tier {tier}: {tier_summary[tier]} variables")
    
    # Define Tabu Edges (forbidden causal directions)
    tabu_edges = []
    for source in labels:
        for target in labels:
            if source == target:
                continue
            source_tier = tiers[source]
            target_tier = tiers[target]
            
            # Forbidden: effect cannot cause its cause (reverse causality)
            if source_tier > target_tier:
                tabu_edges.append((source, target))
    
    print(f"\n[INFO] Running NOTEARS with {len(tabu_edges)} tabu edges...")
    
    # Learn causal structure
    sm = from_pandas(df, tabu_edges=tabu_edges)
    
    # Filter weak edges
    print(f"[INFO] Filtering edges with weight < {threshold}...")
    sm.remove_edges_below_threshold(threshold)
    
    # Convert to NetworkX DiGraph
    G = nx.DiGraph()
    edges_with_weights = []
    
    for u, v, data in sm.edges(data=True):
        weight = data.get('weight', 0)
        G.add_edge(u, v, weight=weight)
        edges_with_weights.append((u, v, weight))
    
    # Ensure DAG by removing cycles
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            weakest_edge = min(cycle, key=lambda e: G[e[0]][e[1]]['weight'])
            print(f"[WARN] Cycle detected. Removing weakest edge: {weakest_edge[0]} -> {weakest_edge[1]}")
            G.remove_edge(*weakest_edge)
        except nx.NetworkXNoCycle:
            break
    
    print(f"\n[INFO] Inferred Causal Edges (weight threshold: {threshold}):")
    top_edges = sorted(G.edges(data=True), key=lambda x: abs(x[2]['weight']), reverse=True)[:20]
    for u, v, weight in top_edges:
        print(f"  {u} -> {v} (weight: {weight['weight']:.4f})")
    if len(G.edges()) > 20:
        print(f"  ... and {len(G.edges()) - 20} more edges")
    
    # Visualize the causal graph
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use hierarchical layout based on tiers
    pos = {}
    tier_nodes = {}
    
    for node in G.nodes():
        tier = tiers[node]
        if tier not in tier_nodes:
            tier_nodes[tier] = []
        tier_nodes[tier].append(node)
    
    # Position nodes by tier
    for tier in sorted(tier_nodes.keys()):
        nodes = tier_nodes[tier]
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - n/2) * 1.5
            y = -tier * 2.5
            pos[node] = (x, y)
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        tier = tiers[node]
        if tier == 0:
            node_colors.append('lightgreen')  # Inputs
        elif tier == 1:
            node_colors.append('lightblue')   # Request rates
        elif tier == 2:
            node_colors.append('plum')        # Node CPU usage
        elif tier == 3:
            node_colors.append('lightyellow') # Service latency
        elif tier == 4:
            node_colors.append('lightcoral')  # E2E latency
        else:
            node_colors.append('lightgray')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, ax=ax)
    
    # Draw labels with smaller font
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
    
    # Draw edges with varying width based on weight
    edges = G.edges()
    weights = [abs(G[u][v]['weight']) for u, v in edges]
    max_weight = max(weights) if weights else 1
    
    edge_widths = [0.5 + 2 * (abs(G[u][v]['weight']) / max_weight) for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', 
                          arrows=True, arrowsize=10, arrowstyle='->', 
                          connectionstyle='arc3,rad=0.1', ax=ax, alpha=0.6)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', label='Tier 0: Inputs (Replicas)'),
        Patch(facecolor='lightblue', label='Tier 1: Request Rates'),
        Patch(facecolor='plum', label='Tier 2: Node CPU Usage'),
        Patch(facecolor='lightyellow', label='Tier 3: Service Latency'),
        Patch(facecolor='lightcoral', label='Tier 4: E2E Latency')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('Causal Graph: Workflow Services Dataset (NOTEARS + Tier Constraints)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('workflow_causal_graph.png', dpi=300, bbox_inches='tight')
    print("\n[INFO] Saved causal graph visualization to workflow_causal_graph.png")
    plt.show()
    
    # Print summary statistics
    print(f"\n[INFO] Graph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")
    print(f"  Connected components: {nx.number_weakly_connected_components(G)}")
    
    return G, df


if __name__ == "__main__":
    G, df = infer_and_visualize_causal_graph("workflow_services_dataset.csv", threshold=0.05)
