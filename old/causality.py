import networkx as nx
import matplotlib.pyplot as plt
from causality import *
from causalnex.structure.notears import from_pandas
from causalnex.network import BayesianNetwork
from sklearn.model_selection import train_test_split
from causalnex.inference import InferenceEngine
from causalnex.structure import StructureModel
from networkx.algorithms.components import weakly_connected_components
import re

def sanitize_string(string):
        # Replace invalid chars with underscore
        safe_name = re.sub(r'[^0-9a-zA-Z_]', '_', string)
        # Avoid double/trailing underscores
        safe_name = re.sub(r'__+', '_', safe_name).strip("_")
        return safe_name

def get_unique_values_dict(df,graph):
    data_def = {col: sorted(df[col].dropna().unique().tolist()) for col in df.columns if sanitize_string(col) in list(graph.nodes)}
    return data_def

def draw_graph(G):
    # Define node positions
    pos = nx.spring_layout(G)  # Positioning for better visualization

    # Draw nodes
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', arrows=True, node_size=2000, font_size=12)
    # Draw edges with weights
    arc_rad = 0.2  # Curve radius for bidirectional edges
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        weight = f"{d['weight']:.4f}"  # Format weight to 4 decimals
        edge_labels[(u, v)] = weight
        #nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f"arc3,rad={arc_rad}", edge_color="black", alpha=0.7, arrows=True, width=2)
        arc_rad = -arc_rad  # Alternate curvature for better visibility

    # Draw edge labels (weights)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.title("Filtered DAG with Strongest Causal Edges & Weights")
    plt.show()


def get_causal_graph(sm):
    graph_data = sm._adj
    edges = []
    for src, targets in graph_data.items():
        for tgt, attr in targets.items():
            edges.append((src, tgt, attr['weight']))

    # build DiGraph
    G = nx.DiGraph()
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    # Ensure acyclic
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        weakest = min(cycle, key=lambda e: G[e[0]][e[1]]['weight'])
        print(f"Removing edge {weakest} to break cycle")
        G.remove_edge(*weakest)

    # Ensure one connected component (keep the largest)
    if nx.number_weakly_connected_components(G) > 1:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Reduced to largest connected component with {len(G.nodes)} nodes")

    draw_graph(G)
    return G

class CausalAnalyser():
    def __init__(self,config,df,cg=None):
        self.config = config
        self.df = df
        if cg is None:
            sm = from_pandas(df)
            if 'threshold' not in config:
                config['threshold'] = 0.1
            sm.remove_edges_below_threshold(config['threshold'])
            self.cg = get_causal_graph(sm)

        self.sm = StructureModel()
        self.sm.add_edges_from(self.cg.edges(data=True))
        self.bn = BayesianNetwork(self.sm)
        self.DATADEF = get_unique_values_dict(df,self.cg)

    def train(self):
        self.bn = self.bn.fit_node_states(self.df)
        self.bn = self.bn.fit_cpds(self.df, method="BayesianEstimator", bayes_prior="K2")
        self.ie = InferenceEngine(self.bn)

    def find_root_cause_causal(self,observation,anovar,anoval,method='interventional'):

        # 1. Abduction (Observing the Evidence)
        baseline = self.ie.query(observation)[anovar][anoval]

        root_cause = None
        max_impact = float('-inf')
        impact_results = {}
        for var in self.DATADEF:
            if var == anovar:
                continue
            impacts = []
            #print(f'Intervening on variable "{var}"')
            for val in [vval for vval in self.DATADEF[var] if vval != observation[var]]:
                cf = observation.copy()
                if method == 'interventional':
                    # 2. Action (Counterfactual Adjustment)
                    intervention = {vv: 0.0 for vv in self.DATADEF[var]}
                    intervention[val] = 1.0
                    try:
                        self.ie.do_intervention(var,intervention)
                    except Exception as e:
                        print(f"Error for variable '{var}'")
                        raise(e)
                    for vvar in [vvvar for vvvar in self.DATADEF if vvvar not in list(self.ie._cpds.keys())]:
                        del cf[vvar] # Delete observations of nodes that were deleted due to the intervention

                    # 3. Prediction (Recomputing the Outcome)
                    p = self.ie.query(cf)[anovar][anoval]
                    
                    self.ie.reset_do(var)
                elif method == 'observational':
                    cf[var] = val
                    p = self.ie.query(cf)[anovar][anoval]
                else:
                    raise ValueError("Invalid method. Choose 'observation' or 'intervention'")
                # 4. Calculate impact - Track the most influential value change
                impact = baseline - p
                impacts.append((val, impact))
                if impact > max_impact:
                    max_impact = impact
                    root_cause = (var, observation[var], val)
            impact_results[var] = impacts

        # Results
        #print(f"Impact of counterfactual changes on P({anovar}={anoval}):")
        likely_causes = []
        for ancestor, impacts in impact_results.items():
            for val, impact in impacts:
                #print(f"P({anovar}={anoval} | do({ancestor}={observation[ancestor]})) - P({anovar}={anoval} | do({ancestor}={val})) = {impact:.4f}")
                if not any(c[0] == ancestor for c in likely_causes):
                    likely_causes.append((ancestor,impact))

        #print(f"\nMost likely root causes:\n")
        #for i,cause in enumerate(likely_causes):
        #    print(f'{i}.'+(4-(1+i//10))*' '+f'{cause[0]}: {cause[1]}')
        return likely_causes