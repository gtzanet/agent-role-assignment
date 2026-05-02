import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from typing import Dict, List, Tuple
from itertools import combinations

from environment import CausalGraph, Node, NodeType
from metrics import calculate_edge_weight

class Allocator:
    def __init__(self, causal_graph: CausalGraph):
        self.causal_graph = causal_graph
        self.tig = nx.Graph() # Task Interaction Graph

    def build_interaction_graph(self, alpha: float = 1.0, beta: float = 0.5, complexity_limit: int = 100):
        """
        Projects the CausalGraph into a Task Interaction Graph (TIG).
        Nodes = tasks (inputs).
        Edge weights = SLA Pull - Complexity Push.
        """
        inputs = self.causal_graph.get_inputs()
        self.tig.clear()
        
        # Add all input nodes to TIG
        for node in inputs:
            self.tig.add_node(node.name, decision_space=node.decision_space_size)
            
        # Calculate weights for all pairs
        for node_a, node_b in combinations(inputs, 2):
            reachable_a = self.causal_graph.get_reachable_kpis(node_a.name)
            reachable_b = self.causal_graph.get_reachable_kpis(node_b.name)
            
            weight = calculate_edge_weight(
                reachable_a=reachable_a,
                reachable_b=reachable_b,
                size_a=node_a.decision_space_size,
                size_b=node_b.decision_space_size,
                alpha=alpha,
                beta=beta,
                complexity_limit=complexity_limit
            )
            
            # Only add edge if there is a positive bond
            if weight > 0:
                self.tig.add_edge(node_a.name, node_b.name, weight=weight)
                
    def partition_tasks(self, n_agents: int) -> Dict[int, List[str]]:
        """
        Partitions the TIG into n_agents clusters using Spectral Clustering.
        Returns a dictionary: {agent_id: [task_names]}
        """
        nodes = list(self.tig.nodes())
        if not nodes:
            return {}
        
        # If fewer nodes than agents, just assign one per agent
        if len(nodes) <= n_agents:
            return {i: [node] for i, node in enumerate(nodes)}

        # Create adjacency matrix based on weights
        adj_matrix = nx.to_numpy_array(self.tig, nodelist=nodes, weight='weight')
        
        # Spectral Clustering
        # using 'precomputed' affinity because we have the weights in adj_matrix
        sc = SpectralClustering(
            n_clusters=n_agents, 
            affinity='precomputed', 
            random_state=42,
            assign_labels='discretize'
        )
        
        labels = sc.fit_predict(adj_matrix)
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(nodes[i])
            
        return clusters

    def get_tig_edges(self) -> List[Tuple[str, str, float]]:
        """Helper to inspect the built graph."""
        return [
            (u, v, d['weight']) 
            for u, v, d in self.tig.edges(data=True)
        ]
