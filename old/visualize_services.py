import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from environment import NodeType


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

def visualize_services_graph(input_file: str, output_file: str, threshold: float = 0.1):
    print(f"[INFO] Reading {input_file}...")
    # Infer Graph
    from causality.causal_discovery import infer_causal_graph_notears
    # Note: threshold is passed to infer_causal_graph
    dataset = pd.read_csv(input_file)
    node_types, decision_space_sizes = _service_dataset_node_metadata(list(dataset.columns))
    causal_graph = infer_causal_graph_notears(
        input_file,
        node_types=node_types,
        threshold=threshold,
        decision_space_sizes=decision_space_sizes,
    )
    
    # Convert CausalGraph to NetworkX for visualization
    # The CausalGraph object has a .graph attribute which is a nx.DiGraph
    # But it might not have the 'type' attributes on nodes needed for coloring.
    # So we copy structure and add attributes.
    
    G_inferred = causal_graph.graph
    G = nx.DiGraph()
    
    # 1. Add Nodes with Types (from CausalGraph)
    for node_name, node_obj in causal_graph.nodes.items():
        # Map NodeType enum to string for color map
        n_type = "intermediary" # default
        if node_obj.node_type.value == "input": n_type = "input"
        elif node_obj.node_type.value == "kpi": n_type = "kpi"
        elif "Load" in node_name: n_type = "load" # refinement for coloring
        elif "CPU" in node_name: n_type = "cpu"   # refinement for coloring
        
        G.add_node(node_name, type=n_type)
        
    # 2. Add Edges (from CausalGraph)
    # infer_causal_graph doesn't store weights in the nx graph by default, 
    # but the edges exist. We can optionally fetch weights from correlation if needed 
    # or just visualize structure.
    # Let's fetch correlation again just for the label since CausalGraph might not strictly store it in edge data.
    
    # Re-calculate correlation just for labels (optional but good for viz)
    df = pd.read_csv(input_file)
    corr = df.corr()
    
    for u, v in G_inferred.edges():
        weight = abs(corr.loc[u, v])
        G.add_edge(u, v, weight=weight, label=f"{corr.loc[u, v]:.2f}")
                
    # Layout (Manual ordering for clarity)
    pos = {}
    
    # Inputs (x=0)
    pos["Replicas_S1"] = (0, 3.5)
    pos["Node_S1"] = (0, 3.0)
    pos["Replicas_S2"] = (0, 1.0)
    pos["Node_S2"] = (0, 0.5)
    
    # Load (x=1)
    pos["Load_N1"] = (1, 2.5)
    pos["Load_N2"] = (1, 1.5)
    
    # CPU (x=2)
    pos["CPU_N1"] = (2, 2.5)
    pos["CPU_N2"] = (2, 1.5)
    
    # KPIs (x=3)
    pos["Lat_S1"] = (3, 2.5)
    pos["Lat_S2"] = (3, 1.5)
    
    plt.figure(figsize=(12, 6))
    
    color_map = {
        'input': 'lightblue', 
        'load': 'lightyellow', 
        'cpu': 'orange', 
        'kpi': 'lightgreen'
    }
    node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, edgecolors='black')
    nx.draw_networkx_labels(G, pos)
    
    if G.edges:
        weights = [G[u][v]['weight'] * 2 for u, v in G.edges]
        nx.draw_networkx_edges(G, pos, width=weights, arrowstyle='->', arrowsize=20)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
    plt.title("Service Refinement (Replicas -> Load -> CPU -> Latency)")
    plt.axis('off')
    
    print(f"[INFO] Saving graph to {output_file}...")
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    visualize_services_graph("services_dataset.csv", "services_graph.png")
