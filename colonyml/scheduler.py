import os
import psutil
from typing import List, Dict


class NodeScheduler:
    """
    Decides how much training work each node gets.

    Faster machines get bigger batches.
    Busy/hot machines get smaller batches.
    Goal: all machines finish at roughly the same time.

    Usage:
        scheduler = NodeScheduler()
        assignments = scheduler.assign_batch_sizes(nodes, total_batch=256)
    """

    def get_local_metrics(self) -> dict:
        """Get this machine's current performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        ram = psutil.virtual_memory()
        cpu_cores = os.cpu_count() or 1

        return {
            "cores": cpu_cores,
            "cpu_usage": cpu_percent,
            "cpu_freq_mhz": cpu_freq.current if cpu_freq else 2000,
            "ram_available_gb": ram.available / (1024 ** 3),
            "ram_total_gb": ram.total / (1024 ** 3)
        }

    def compute_node_weight(self, metrics: dict) -> float:
        """
        Calculate how much work this node should get.
        Higher weight = more work assigned.

        Considers:
        - Number of CPU cores
        - Current CPU usage (busy = less weight)
        - Available RAM
        - CPU frequency
        """
        cores = metrics["cores"]
        usage = metrics["cpu_usage"] / 100.0
        freq = metrics["cpu_freq_mhz"] / 3000.0
        ram_factor = min(
            metrics["ram_available_gb"] / 8.0, 1.0
        )

        base_weight = cores * freq
        availability = 1.0 - (usage * 0.5)
        weight = base_weight * availability * ram_factor

        return max(0.1, weight)

    def assign_batch_sizes(
        self,
        nodes: List[dict],
        total_batch_size: int
    ) -> Dict[str, int]:
        """
        Given nodes and total batch size,
        return how many samples each node processes.
        """
        if not nodes:
            return {}

        # Get weight for each node
        weights = {}
        for node in nodes:
            metrics = node.get(
                "metrics", self.get_local_metrics()
            )
            weights[node["ip"]] = self.compute_node_weight(
                metrics
            )

        total_weight = sum(weights.values())

        # Assign proportional batch sizes
        assignments = {}
        assigned = 0

        for i, node in enumerate(nodes):
            ip = node["ip"]

            if i == len(nodes) - 1:
                # Last node gets remainder
                assignments[ip] = total_batch_size - assigned
            else:
                share = weights[ip] / total_weight
                batch = max(1, int(total_batch_size * share))
                assignments[ip] = batch
                assigned += batch

        return assignments

    def print_assignment_table(
        self,
        nodes: List[dict],
        assignments: Dict[str, int]
    ):
        """Pretty print the batch assignment."""
        total = sum(assignments.values())

        print("\nBatch Assignment")
        print("=" * 65)
        print(
            f"{'Node':<20} {'IP':<15} {'Cores':>6} "
            f"{'CPU%':>6} {'Batch':>8} {'Share%':>8}"
        )
        print("-" * 65)

        for node in nodes:
            ip = node["ip"]
            metrics = node.get(
                "metrics", self.get_local_metrics()
            )
            batch = assignments.get(ip, 0)
            share = (batch / total * 100) if total > 0 else 0

            print(
                f"{node.get('hostname', 'unknown'):<20} "
                f"{ip:<15} "
                f"{metrics.get('cores', '?'):>6} "
                f"{metrics.get('cpu_usage', 0):>5.1f}% "
                f"{batch:>8} "
                f"{share:>7.1f}%"
            )

        print("=" * 65)
        print(f"{'Total batch size:':>50} {total}")


if __name__ == "__main__":
    scheduler = NodeScheduler()

    print("Local Machine Metrics:")
    metrics = scheduler.get_local_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Simulate two nodes with different specs
    nodes = [
        {
            "ip": "192.168.1.100",
            "hostname": "powerful-machine",
            "metrics": {
                "cores": 16,
                "cpu_usage": 20.0,
                "cpu_freq_mhz": 3500,
                "ram_available_gb": 14.0,
                "ram_total_gb": 16.0
            }
        },
        {
            "ip": "192.168.1.101",
            "hostname": "Jarvis",
            "metrics": {
                "cores": 14,
                "cpu_usage": 35.0,
                "cpu_freq_mhz": 2400,
                "ram_available_gb": 10.0,
                "ram_total_gb": 16.0
            }
        }
    ]

    print("\nSimulated 2-node cluster:")
    assignments = scheduler.assign_batch_sizes(nodes, 256)
    scheduler.print_assignment_table(nodes, assignments)

    print("\nSimulated 3-node cluster:")
    nodes.append({
        "ip": "192.168.1.102",
        "hostname": "weak-laptop",
        "metrics": {
            "cores": 4,
            "cpu_usage": 70.0,
            "cpu_freq_mhz": 1800,
            "ram_available_gb": 2.0,
            "ram_total_gb": 8.0
        }
    })
    assignments = scheduler.assign_batch_sizes(nodes, 256)
    scheduler.print_assignment_table(nodes, assignments)