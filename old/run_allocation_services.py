from environment import CausalGraph, Node, NodeType
from allocation import Allocator


def _service_dataset_node_metadata(columns: list[str]) -> tuple[dict[str, NodeType], dict[str, int]]:
    node_types: dict[str, NodeType] = {}
    decision_space_sizes: dict[str, int] = {}

    for name in columns:
        if name.startswith("Replicas_"):
            node_types[name] = NodeType.INPUT
            decision_space_sizes[name] = 10
        elif name.startswith("Node_"):
            node_types[name] = NodeType.INPUT
            decision_space_sizes[name] = 2
        elif name.startswith("Lat_"):
            node_types[name] = NodeType.KPI
        else:
            node_types[name] = NodeType.INTERMEDIARY

    return node_types, decision_space_sizes

def run_allocation():
    print("=== Running Service Allocation Optimization ===")
    
    # 1. Infer Causal Graph from Dataset
    from causality.causal_discovery import infer_causal_graph_notears
    # Using correlation threshold 0.1 to capture weaker links like Replicas_S2 -> Load_N2 if needed
    dataset = pd.read_csv("services_dataset.csv")
    node_types, decision_space_sizes = _service_dataset_node_metadata(list(dataset.columns))
    graph = infer_causal_graph_notears(
        "services_dataset.csv",
        node_types=node_types,
        threshold=0.1,
        decision_space_sizes=decision_space_sizes,
    )
    
    # Verify inputs are correct (for Allocator)
    # Ensure decision_space_size is set (handled in infer_causal_graph)
    
    # 3. Build Allocation Map
    allocator = Allocator(graph)
    
    # Parameters
    # alpha=1.0 
    # beta=0.5 (Lowered from 2.0 to allow forming bonds between Replicas+Node)
    # complexity_limit=50
    
    print("\nBuilding Interaction Graph...")
    allocator.build_interaction_graph(alpha=1.0, beta=0.5, complexity_limit=50)
    
    print("\nCalculated Edge Weights:")
    for u, v, w in allocator.get_tig_edges():
        print(f"  {u} -- {v} : {w:.4f}")
        
    # Visualization of TIG
    import matplotlib.pyplot as plt
    import networkx as nx
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(allocator.tig, seed=42)
    
    # Draw Nodes
    nx.draw_networkx_nodes(allocator.tig, pos, node_size=3000, node_color='lightgreen', edgecolors='black')
    nx.draw_networkx_labels(allocator.tig, pos)
    
    # Draw Edges
    edges = allocator.tig.edges(data=True)
    if edges:
        weights = [d['weight'] * 5 for u, v, d in edges]
        nx.draw_networkx_edges(allocator.tig, pos, width=weights)
        edge_labels = { (u, v): f"{d['weight']:.2f}" for u, v, d in edges }
        nx.draw_networkx_edge_labels(allocator.tig, pos, edge_labels=edge_labels)
        
    plt.title("Task Interaction Graph (TIG)")
    plt.axis('off')
    output_file = "tig_services.png"
    plt.savefig(output_file)
    print(f"\n[INFO] Saved TIG visualization to {output_file}")
    plt.close()

    print("\nPartitioning into 2 Agents...")
    clusters = allocator.partition_tasks(n_agents=2)
    
    for agent_id, tasks in clusters.items():
        print(f"\n[Agent {agent_id}]")
        for t in tasks:
            print(f"  - {t}")
            
    # Verification
    # We expect [Replicas_S1, Node_S1] to be together (Size 20)
    # We expect [Replicas_S2, Node_S2] to be together (Size 20)
    # We expect them to be separated because joining Replicas_S1 + Replicas_S2 = Size 100 (Penalty!)

if __name__ == "__main__":
    run_allocation()
