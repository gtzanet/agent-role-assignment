from enum import Enum
import networkx as nx
from typing import List, Set, Dict

class NodeType(Enum):
    INPUT = "input"
    INTERMEDIARY = "intermediary"
    KPI = "kpi"

class Node:
    def __init__(self, name: str, node_type: NodeType, decision_space_size: int = 1):
        """
        Args:
            name: Unique identifier for the node.
            node_type: Role of the node in the system.
            decision_space_size: For INPUT nodes, the number of possible decision options.
                                 Defaults to 1 for non-inputs.
        """
        self.name = name
        self.node_type = node_type
        self.decision_space_size = decision_space_size

    def __repr__(self):
        return f"Node({self.name}, {self.node_type.value}, size={self.decision_space_size})"

class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node):
        self.nodes[node.name] = node
        self.graph.add_node(node.name, data=node)

    def add_edge(self, source_name: str, target_name: str):
        if source_name not in self.nodes or target_name not in self.nodes:
            raise ValueError("Both nodes must be added to the graph before creating an edge.")
        self.graph.add_edge(source_name, target_name)

    def get_inputs(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]

    def get_kpis(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.KPI]

    def get_reachable_kpis(self, input_node_name: str) -> Set[str]:
        """
        Finds all KPI nodes reachable from a given input node.
        """
        if input_node_name not in self.nodes:
            raise ValueError(f"Node {input_node_name} not found.")
        
        # Get all descendants in the DAG
        descendants = nx.descendants(self.graph, input_node_name)
        
        # Filter for KPIs
        reachable_kpis = {
            node_name for node_name in descendants 
            if self.nodes[node_name].node_type == NodeType.KPI
        }
        
        return reachable_kpis
