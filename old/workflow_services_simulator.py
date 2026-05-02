import numpy as np
from typing import Dict, List, Tuple

class WorkflowServiceSimulator:
    """
    Simulates a Workflow-based Service Deployment environment.
    
    Structure:
    - Multiple Workflows, each a chain of services: S1 -> S2 -> S3 -> ...
    - Each service can have multiple replicas
    - Each service is statically placed on a node
    - Arrival rate of a service depends on the service rate of the caller
    
    For each service sample, we record:
    - Workflow ID
    - Service ID
    - Replicas: Number of service replicas (1-10)
    - Node: Statically assigned node ID
    - Arrival Rate: Requests per second arriving at this service
    - Service Rate: Requests per second that service can handle (per replica)
    - Latency: Average latency for this service (ms)
    - E2E Latency: End-to-end latency of the entire workflow
    """
    
    def __init__(self, n_workflows: int = 2, max_chain_length: int = 2, n_nodes: int = 2, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_workflows = n_workflows
        self.max_chain_length = max_chain_length
        self.n_nodes = n_nodes
        
        # Generate static workflow structure
        self.workflows = self._generate_workflows()
        
    def _generate_workflows(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate workflow structures.
        Each workflow is a list of (service_id, node_id) tuples.
        service_id is unique across all workflows.
        Returns dict: {workflow_id: [(service_id, node_id), ...]}
        """
        workflows = {}
        service_id = 0
        
        for wf_id in range(self.n_workflows):
            chain_length = self.rng.integers(2, self.max_chain_length + 1)
            workflow_services = []
            
            for _ in range(chain_length):
                node_id = self.rng.integers(0, self.n_nodes)
                workflow_services.append((service_id, node_id))
                service_id += 1
            
            workflows[wf_id] = workflow_services
        
        return workflows
    
    def generate_sample(self) -> Dict[str, any]:
        """
        Generate a single sample with all services from all workflows.
        Returns a list of dicts, one per service.
        Node is used internally for interdependencies but not exported.
        """
        samples = []
        workflow_indices = {}

        # Track latency by node for interdependency while sampling services
        node_latencies = {}

        for wf_id, services in self.workflows.items():
            service_rates = {}
            arrival_rates = {}
            wf_start_idx = len(samples)

            # Base arrival for first service in workflow
            base_arrival_rate = self.rng.uniform(10, 100)
            arrival_rates[services[0][0]] = base_arrival_rate

            for idx, (service_id, node_id) in enumerate(services):
                replicas = self.rng.integers(1, 11)

                service_rate_per_replica = self.rng.uniform(20, 200)
                total_service_rate = service_rate_per_replica * replicas
                service_rates[service_id] = total_service_rate

                if idx == 0:
                    arr_rate = arrival_rates[service_id]
                else:
                    prev_service_id = services[idx - 1][0]
                    arr_rate = min(
                        arrival_rates[prev_service_id],
                        service_rates[prev_service_id]
                    )
                    arrival_rates[service_id] = arr_rate

                utilization = min(arr_rate / total_service_rate, 0.99)
                base_latency = 5.0
                latency = base_latency + (50 * (utilization ** 2)) + self.rng.normal(0, 0.5)

                # Interdependency from co-located services
                if node_id in node_latencies and len(node_latencies[node_id]) > 0:
                    latency += 0.2 * np.mean(node_latencies[node_id])

                latency = max(base_latency, latency)
                node_latencies.setdefault(node_id, []).append(latency)

                samples.append({
                    "Workflow_ID": wf_id,
                    "Service_ID": service_id,
                    "Replicas": replicas,
                    "Arrival_Rate": arr_rate,
                    "Service_Rate": total_service_rate,
                    "Latency_ms": latency,
                    "_Node_ID": node_id,
                })

            workflow_indices[wf_id] = (wf_start_idx, len(samples))

        # Compute per-node CPU usage from aggregated utilization
        node_utilization_sum = {}
        for s in samples:
            node_id = s["_Node_ID"]
            util = min(s["Arrival_Rate"] / s["Service_Rate"], 1.5)
            node_utilization_sum[node_id] = node_utilization_sum.get(node_id, 0.0) + util

        node_cpu = {}
        for node_id, util_sum in node_utilization_sum.items():
            cpu = 0.1 + 0.25 * util_sum + self.rng.normal(0, 0.02)
            node_cpu[node_id] = float(np.clip(cpu, 0.0, 1.0))

        # Attach node CPU to each service and add CPU impact on service latency
        for s in samples:
            cpu = node_cpu[s["_Node_ID"]]
            s["Node_CPU_Usage"] = cpu
            s["Latency_ms"] = max(5.0, s["Latency_ms"] + 20.0 * (cpu ** 2))

        # Compute end-to-end latency per workflow after final latency is set
        for wf_id, (start_idx, end_idx) in workflow_indices.items():
            e2e_latency = sum(s["Latency_ms"] for s in samples[start_idx:end_idx])
            for i in range(start_idx, end_idx):
                samples[i]["E2E_Latency_ms"] = e2e_latency
                samples[i].pop("_Node_ID", None)

        return samples
    
    def generate_batch(self, n_samples: int) -> Dict[str, list]:
        """
        Generate n_samples, each with all services flattened into a single row.
        Returns dict with keys like Service_0_Replicas, Service_0_Arrival_Rate, etc.
        """
        all_data = []
        
        for _ in range(n_samples):
            sample_data = {}
            services_by_wf = self.generate_sample()
            
            # Flatten the nested structure: group by service
            services_dict = {}
            workflow_e2e = {}
            
            for item in services_by_wf:
                service_id = item['Service_ID']
                workflow_id = item['Workflow_ID']
                e2e_latency = item['E2E_Latency_ms']
                
                services_dict[service_id] = item
                workflow_e2e[workflow_id] = e2e_latency
            
            # Create flattened columns for each service
            for service_id in sorted(services_dict.keys()):
                service_data = services_dict[service_id].copy()
                service_data.pop('Service_ID')  # Remove redundant ID
                service_data.pop('Workflow_ID')  # Remove workflow ID from service data
                service_data.pop('E2E_Latency_ms')  # Handle E2E latency separately
                
                for key, value in service_data.items():
                    col_name = f"Service_{service_id}_{key}"
                    sample_data[col_name] = value
            
            # Add E2E latencies per workflow
            for wf_id in sorted(workflow_e2e.keys()):
                col_name = f"E2E_Latency_Workflow_{wf_id}"
                sample_data[col_name] = workflow_e2e[wf_id]
            
            all_data.append(sample_data)
        
        return all_data


if __name__ == "__main__":
    print("=== Workflow Services Simulator ===")
    sim = WorkflowServiceSimulator(n_workflows=2, max_chain_length=2, n_nodes=2)
    
    # Show workflow structure
    print("\nWorkflow Structure:")
    for wf_id, services in sim.workflows.items():
        service_info = " -> ".join([f"S{sid}(N{nid})" for sid, nid in services])
        print(f"  Workflow {wf_id}: {service_info}")
    
    print("\n=== Generating a single sample ===")
    sample = sim.generate_sample()
    for service_data in sample:
        print(f"  WF{service_data['Workflow_ID']} Service {service_data['Service_ID']}: "
              f"Replicas={service_data['Replicas']}, "
              f"Arrival={service_data['Arrival_Rate']:.2f} req/s, "
              f"Service={service_data['Service_Rate']:.2f} req/s, "
              f"NodeCPU={service_data['Node_CPU_Usage']:.3f}, "
              f"Latency={service_data['Latency_ms']:.2f}ms, "
              f"E2E={service_data['E2E_Latency_ms']:.2f}ms")
