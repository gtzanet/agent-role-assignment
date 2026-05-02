
import pandas as pd

from environment import NodeType
from causality.causal_discovery import infer_causal_graph_notears


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

def main():
    print("=== Verifying Causal Inference ===")
    try:
        dataset = pd.read_csv("services_dataset.csv")
        node_types, decision_space_sizes = _service_dataset_node_metadata(list(dataset.columns))
        graph = infer_causal_graph_notears(
            "services_dataset.csv",
            node_types=node_types,
            decision_space_sizes=decision_space_sizes,
        )
        
        print("\n[INFO] Graph Structure Verification:")
        inputs = graph.get_inputs()
        kpis = graph.get_kpis()
        
        print(f"Inputs Found: {[n.name for n in inputs]}")
        print(f"KPIs Found: {[n.name for n in kpis]}")
        
        # Check specific expected edges (Ground Truth Check)
        expected_edges = [
            ("Replicas_S1", "Load_N1"),
            ("Replicas_S2", "Load_N2"),
            ("Load_N1", "CPU_N1"),
            ("CPU_N1", "Lat_S1")
        ]
        
        print("\n[INFO] Checking specific expected edges:")
        # CausalGraph stores the nx.DiGraph in .graph
        current_edges = list(graph.graph.edges())
        
        for u, v in expected_edges:
            if (u, v) in current_edges:
                print(f"  [PASS] {u} -> {v} exists.")
            else:
                # Check for reverse edge as LiNGAM might flip it
                if (v, u) in current_edges:
                     print(f"  [WARN] {u} -> {v} NOT found directly. But {v} -> {u} EXISTS (Reverse Causal Direction inferred).")
                else:
                    print(f"  [WARN] {u} -> {v} NOT found directly.")
                
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    main()
