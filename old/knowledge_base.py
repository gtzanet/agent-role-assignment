import networkx as nx
from matplotlib import pyplot as plt
import json
import pandas as pd
import numpy as np

def get_unique_values_dict(df):
    data_def = {col: sorted(df[col].dropna().unique().tolist()) for col in df.columns}
    return data_def

def discretize_dataframe(df, bins=5):
    discretized_df = pd.DataFrame()
    mappings = {}

    for col in df.columns:
        unique_values = df[col][df[col] != -1].unique()
        min_val, max_val = df[col][df[col] != -1].min(), df[col][df[col] != -1].max()
        
        if np.array_equal(unique_values, unique_values.astype(int)):  # Discrete values
            bins = max_val - min_val + 2
            bin_edges = list(range(min_val,max_val+2))
        else:  # Continuous values
            bins = bins
            bin_edges = np.linspace(0.9 * min_val, 1.1 * max_val, bins)
        bin_edges = np.insert(bin_edges,0,-1)
        labels = range(bins)

        discretized_df[col] = pd.cut(df[col], bins=bin_edges, labels=labels, right=False, include_lowest=True)
        mappings[col] = dict(zip(labels, bin_edges[:-1]))

    return discretized_df, mappings

class Metric:
    def __init__(self, type, name):
        self.type = type
        self.name = name
    
    def __repr__(self):
        return f"Metric(type={self.type}, name={self.name})"
    
    def to_dict(self):
        return {"type": self.type, "name": self.name}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["type"], data["name"])


class Node:
    def __init__(self, name, cluster, ip, connections):
        self.name = name
        self.cluster = cluster
        self.ip = ip
        self.connections = connections
    
    def __repr__(self):
        return f"Node(name={self.name}, cluster={self.cluster}, ip={self.ip}, connections={self.connections})"
    
    def to_dict(self):
        return {"name": self.name, "cluster": self.cluster, "IP": self.ip, "connections": self.connections}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["cluster"], data["IP"], data["connections"])


class Application:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"Application(name={self.name})"
    
    def to_dict(self):
        return {"name": self.name}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"])


class Service:
    def __init__(self, name, parent, comm, application):
        self.name = name
        self.parent = parent
        self.comm = comm
        self.application = application
    
    def __repr__(self):
        return f"Service(name={self.name}, parent={self.parent}, comm={self.comm}, application={self.application})"
    
    def to_dict(self):
        return {"name": self.name, "parent": self.parent, "comm": self.comm, "application": self.application.to_dict()}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["parent"], data["comm"], Application.from_dict({"name": data["application"]}))


class DeploymentPlan:
    def __init__(self, service, cluster, node, replicas):
        self.service = service
        self.cluster = cluster
        self.node = node
        self.replicas = replicas
    
    def __repr__(self):
        return f"DeploymentPlan(app_name={self.service}, cluster={self.cluster}, node={self.node}, replicas={self.replicas})"
    
    def to_dict(self):
        return {"app_name": self.service, "cluster": self.cluster, "node": self.node, "replicas": self.replicas}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["service"], data["cluster"], data["node"], data["replicas"])


class Infrastructure:
    def __init__(self, nodes):
        self.nodes = {node.name: node for node in nodes}
    
    def get_cluster_ip(self, cluster):
        for node in self.nodes.values():
            if node.cluster == cluster:
                return node.ip
        return None
    
    def get_node_name_from_ip(self, ip):
        for node in self.nodes.values():
            if node.ip == ip:
                return node.name
        return None
    
    def to_dict(self):
        return {"nodes": [node.to_dict() for node in self.nodes.values()]}
    
    @classmethod
    def from_dict(cls, data):
        nodes = [Node.from_dict(node_data) for node_data in data]
        return cls(nodes)

class KnowledgeBase():
    def __init__(self,data_model=None,data_model_file=None):
        if data_model is None:
            if data_model_file is None:
                 with open("data_model.json") as fp:
                    data_model = json.load(fp)
            else:
                with open(data_model_file) as fp:
                    data_model = json.load(fp)
        self.METRICS = [Metric.from_dict(m) for m in data_model["METRICS"]]
        self.Infrastructure = Infrastructure.from_dict([n for n in data_model["INFRA"]])
        self.SERVICES = [Service.from_dict(s) for s in data_model["SERVICES"]]
        self.DPLAN = [DeploymentPlan.from_dict(p) for p in data_model["DPLAN"]]

    def generate_causal_graph(self,draw=False):
        metric_types = ["network","compute","service","slink"]
        METRICS = json.loads(json.dumps({mtype: [m.name for m in self.METRICS if m.type == mtype] for mtype in metric_types}).replace("-", "_"))
        INFRA = json.loads(json.dumps([node.to_dict() for node in self.Infrastructure.nodes.values()]).replace("-", "_"))
        APP = json.loads(json.dumps({service.name: service.to_dict() for service in self.SERVICES}).replace("-", "_"))
        DPLAN = json.loads(json.dumps({plan.service: plan.to_dict() for plan in self.DPLAN}).replace("-", "_"))
        G = nx.DiGraph()
        self.node_colors = {}
        
        # Mapping of services to their nodes
        service_to_node = {service: DPLAN[service]['node'] for service in DPLAN}
        
        # Create virtual link metrics between compute nodes only for nodes in DPLAN
        relevant_nodes = {DPLAN[service]['node'] for service in DPLAN}
        vlinks = {}
        for node1 in INFRA:
            if node1['name'] in relevant_nodes:
                for node2_ip in node1['connections']:
                    node2 = next((n for n in INFRA if n['IP'] == node2_ip), None)
                    if node2 and node2['name'] in relevant_nodes and node1['name'] != node2['name']:
                        vlink = f"{node1['name']}_{node2['name']}_vlink"
                        vlinks[vlink] = []
                        metric_node_status = f"{vlink}_{'node_status'}"
                        vlinks[vlink].append(metric_node_status)
                        self.node_colors[metric_node_status] = 'red'
                        
                        for metric in METRICS['network']:
                            metric_node = f"{vlink}_{metric}"
                            vlinks[vlink].append(metric_node)
                            self.node_colors[metric_node] = 'red'
        
        # Add edges based on arrival rate effects
        for service, plan in DPLAN.items():
            node = plan['node']
            arrival_rate = f"{service}_arrival_rate"
            service_rate = f"{service}_service_rate"
            node_status = f"{node}_node_status"
            replicas = f"{service}_replicas"
            
            G.add_edge(arrival_rate, node_status)
            self.node_colors[arrival_rate] = 'blue'  # Service metrics
            self.node_colors[service_rate] = 'blue'
            self.node_colors[replicas] = 'blue'  # Replicas same color as arrival rate and service rate
            self.node_colors[node_status] = 'green'  # Compute node metrics
            
            # Add replicas node and its effects
            G.add_edge(replicas, node_status)
            for metric in METRICS['compute']:
                metric_node = f"{node}_{metric}"
                G.add_edge(replicas, metric_node)
                if metric != 'node_status':
                    G.add_edge(node_status, metric_node)
                G.add_edge(arrival_rate, metric_node)
                
                # Compute metrics affect the service rate of the services deployed on the node
                G.add_edge(metric_node, service_rate)
                self.node_colors[metric_node] = 'green'
            
            G.add_edge(replicas, service_rate)
            G.add_edge(node_status, service_rate)
            G.add_edge(arrival_rate, service_rate)
        
        # Add edges based on service rate effects with slink integration
        for service, app_info in APP.items():
            if app_info['parent']:
                parent_service = app_info['parent']
                parent_service_rate = f"{parent_service}_service_rate"
                child_arrival_rate = f"{service}_arrival_rate"
                slink_status = f"{parent_service}_to_{service}_slink_status"
                
                G.add_edge(parent_service_rate, slink_status)
                self.node_colors[slink_status] = 'purple'  # Slink nodes
                
                for metric in METRICS['slink']:
                    metric_node = f"{parent_service}_to_{service}_{metric}"
                    G.add_edge(parent_service_rate, metric_node)
                    if metric != 'slink_status':
                        G.add_edge(slink_status, metric_node)
                    G.add_edge(metric_node, child_arrival_rate)
                    self.node_colors[metric_node] = 'purple'
                
                parent_node = service_to_node[parent_service]
                child_node = service_to_node[service]
                vlink = f"{parent_node}_{child_node}_vlink"
                if vlink in vlinks:
                    vlink_metric_node_status = f"{parent_node}_{child_node}_vlink_{'node_status'}"
                    for metric_node in vlinks[vlink]:
                        if metric_node != vlink_metric_node_status:
                            G.add_edge(vlink_metric_node_status, metric_node)
                        for slink_metric in METRICS['slink']:
                            slink_metric_node = f"{parent_service}_to_{service}_{slink_metric}"
                            G.add_edge(metric_node, slink_metric_node)
                    for metric_node in vlinks[vlink]:
                        G.add_edge(parent_service_rate, metric_node)
                G.add_edge(parent_service_rate, child_arrival_rate)
        self.cg = G
        if draw:
            self.draw_causal_graph()

    def draw_causal_graph(self):
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.kamada_kawai_layout(self.cg)  # Improved layout for better visibility
        colors = [self.node_colors.get(node, 'gray') for node in self.cg.nodes()]
        nx.draw(self.cg, pos, with_labels=True, node_size=3000, node_color=colors, font_size=10, edge_color='gray')
        plt.title("Causal Graph")
        plt.show()

    def translate_payload(self,payload):
        metric_types = ["network","compute","service","slink"]
        METRICS = json.loads(json.dumps({mtype: [m.name for m in self.METRICS if m.type == mtype] for mtype in metric_types}).replace("-", "_"))
        INFRA = json.loads(json.dumps([node.to_dict() for node in self.Infrastructure.nodes.values()]).replace("-", "_"))
        APP = json.loads(json.dumps({service.name: service.to_dict() for service in self.SERVICES}).replace("-", "_"))
        data = {"event": payload["event"]}
        observations = payload['observations']
        for s in APP:            
            for metric in METRICS["service"]:
                if s not in observations['services']:
                    data[f'{s}_{metric}'] = -1
                elif metric not in observations['services'][s]:
                    data[f'{s}_{metric}'] = -1
                else:
                    data[f'{s}_{metric}'] = observations['services'][s][metric] if 'cluster' != metric else int(observations['services'][s][metric].lstrip('member'))
        for node in INFRA:
            for metric in METRICS["compute"]:
                data[f'{node["name"]}_{metric}'] = -1
                for cluster in observations["clusters"]:
                    if node["IP"] in observations["clusters"][cluster]:
                        if metric in observations['clusters'][cluster][node["IP"]]:
                            data[f'{node["name"]}_{metric}'] = observations['clusters'][cluster][node["IP"]][metric]
        for dst_svc in [s for s in APP if APP[s]["parent"] is not None]:
            #src_svc,dst_svc = extract_tuple(slink)
            src_svc = APP[dst_svc]["parent"]
            slink_string = f'(\"{src_svc}\", \"{dst_svc}\")'
            for metric in METRICS["slink"]:
                if slink_string not in observations["slinks"]:
                    data[f'{src_svc}_to_{dst_svc}_{metric}'] = -1
                elif metric not in observations["slinks"][slink_string]:
                    data[f'{src_svc}_to_{dst_svc}_{metric}'] = -1
                else:
                    data[f'{src_svc}_to_{dst_svc}_{metric}'] = observations["slinks"][slink_string][metric]
        for node in INFRA:
            for dstIP in node["connections"]:
                for metric in METRICS["network"]:
                    link_string = f'(\"{node["IP"]}\", \"{dstIP}\")'
                    if link_string not in observations['vlinks']:
                        data[f'{node["name"]}_{self.Infrastructure.get_node_name_from_ip(dstIP)}_vlink_{metric}'] = -1
                    else:
                        if metric in observations['vlinks'][link_string]:
                            data[f'{node["name"]}_{self.Infrastructure.get_node_name_from_ip(dstIP)}_vlink_{metric}'] = observations['vlinks'][link_string][metric]
                        else:
                            data[f'{node["name"]}_{self.Infrastructure.get_node_name_from_ip(dstIP)}_vlink_{metric}'] = -1
        return data

    def prepare_dataset(self,observations):
        self.df = pd.DataFrame(observations)
        self.df, self.bin_edges_dict = discretize_dataframe(self.df, bins=4)
        self.df = self.df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric first
        self.df = self.df.astype('int64')  # Convert all columns to Int64
        graph_nodes = set(node for edge in self.cg.edges for node in edge)
        self.df =  self.df[[col for col in self.df.columns if col in graph_nodes]]
        self.datadef = get_unique_values_dict(self.df)
        return self.df,self.datadef

    def translate_bin_to_range(self,bin_value,column):
        """
        Translates a given discretized bin value to its corresponding range in the original space.

        Parameters:
        - bin_value: The discretized bin index.
        - bin_edges: The edges used during discretization.

        Returns:
        - A tuple representing the (min, max) range of the original values for that bin.
        """
        if self.bin_edges_dict[column] is None:
            raise ValueError("bin_edges cannot be None. Ensure discretization has been performed correctly.")

        if not (0 <= bin_value < len(self.bin_edges_dict[column]) - 1):
            raise ValueError(f"Invalid bin value: {bin_value}. It should be in range [0, {len(self.bin_edges_dict[column]) - 2}].")

        return self.bin_edges_dict[column][bin_value], self.bin_edges_dict[column][bin_value + 1]