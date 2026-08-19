"""
ColonyML Trainer — Zero-config distributed training coordinator.

Handles:
1. Auto-discovery of all nodes on the network
2. Automatic rank assignment (lowest IP = rank 0)
3. Barrier synchronization (all nodes start together)
4. Data splitting across nodes
5. Ring AllReduce gradient averaging
"""

import socket
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import List, Callable, Optional
from colonyml.discovery import NodeDiscovery
from colonyml.communicator import NodeCommunicator
from colonyml.compressor import GradientCompressor, CompressionLevel
from colonyml.scheduler import NodeScheduler


class ColonyTrainer:
    """
    Zero-config distributed training coordinator.

    Usage:
        trainer = ColonyTrainer(wait=10)
        trainer.train(
            model=model,
            dataset=dataset,
            epochs=3,
            batch_size=64
        )
    """

    BARRIER_PORT = 29502

    def __init__(
        self,
        port: int = 29500,
        wait: int = 10
    ):
        self.port = port
        self.wait = wait
        self.local_ip = self._get_local_ip()
        self.discovery = NodeDiscovery(port=port)
        self.communicator = NodeCommunicator(
            local_ip=self.local_ip,
            port=port
        )
        self.compressor = GradientCompressor()
        self.scheduler = NodeScheduler()

    def _get_local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()

    def discover_cluster(self) -> List[dict]:
        """
        Announce this node and wait for others to join.
        Returns sorted list of all nodes (including self).
        Lowest IP automatically becomes rank 0.
        """
        print(f"[ColonyML] Announcing node: {self.local_ip}")
        self.discovery.announce()
        self.discovery.start_listening()

        print(f"[ColonyML] Waiting {self.wait}s "
              f"for other nodes to join...")

        for remaining in range(self.wait, 0, -1):
            nodes = self.discovery.get_all_nodes()
            print(
                f"\r[ColonyML] {remaining}s remaining "
                f"| {len(nodes)} node(s) found",
                end="", flush=True
            )
            time.sleep(1)

        print()

        nodes = self.discovery.get_all_nodes()
        nodes = sorted(nodes, key=lambda x: x["ip"])

        print(f"\n[ColonyML] Cluster ready — {len(nodes)} node(s):")
        for i, node in enumerate(nodes):
            tag = "(you)" if node.get("is_self") else ""
            role = "RANK 0 (master)" if i == 0 else f"RANK {i}"
            print(f"  {role}: {node['hostname']} "
                  f"({node['ip']}) {tag}")

        return nodes

    def get_my_rank(self, nodes: List[dict]) -> int:
        """Get this node's rank based on IP ordering."""
        for i, node in enumerate(nodes):
            if node["ip"] == self.local_ip:
                return i
        return 0

    def barrier_sync(
        self,
        nodes: List[dict],
        my_rank: int,
        barrier_name: str = "start"
    ):
        """
        Wait until ALL nodes are ready before proceeding.
        Rank 0 collects ready signals from all others,
        then broadcasts GO signal to everyone.
        """
        n = len(nodes)
        if n == 1:
            print(f"[ColonyML] Single node — no barrier needed")
            return

        print(f"[ColonyML] Barrier sync: {barrier_name}")

        if my_rank == 0:
            # Rank 0: collect READY from all other nodes
            ready_count = 0
            server = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            server.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            server.bind((self.local_ip, self.BARRIER_PORT))
            server.listen(n - 1)
            server.settimeout(60)

            while ready_count < n - 1:
                try:
                    conn, addr = server.accept()
                    msg = conn.recv(1024).decode()
                    if msg == "READY":
                        ready_count += 1
                        print(f"[ColonyML] {ready_count}/{n-1} "
                              f"nodes ready")
                    conn.close()
                except socket.timeout:
                    break

            server.close()

            # Send GO to all other nodes
            for node in nodes[1:]:
                try:
                    with socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM
                    ) as s:
                        s.settimeout(10)
                        s.connect((
                            node["ip"],
                            self.BARRIER_PORT + 1
                        ))
                        s.sendall(b"GO")
                except Exception as e:
                    print(f"[ColonyML] Warning: "
                          f"Could not send GO to {node['ip']}: {e}")

        else:
            # Other ranks: send READY to rank 0, wait for GO
            rank0 = nodes[0]
            time.sleep(1)

            try:
                with socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                ) as s:
                    s.settimeout(10)
                    s.connect((rank0["ip"], self.BARRIER_PORT))
                    s.sendall(b"READY")
            except Exception as e:
                print(f"[ColonyML] Warning: "
                      f"Could not send READY: {e}")

            # Wait for GO
            server = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            server.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            server.bind((self.local_ip, self.BARRIER_PORT + 1))
            server.listen(1)
            server.settimeout(60)

            try:
                conn, _ = server.accept()
                msg = conn.recv(1024).decode()
                conn.close()
            except socket.timeout:
                print("[ColonyML] Warning: Barrier timeout")
            finally:
                server.close()

        print(f"[ColonyML] All nodes synchronized — starting!")

    def get_data_subset(
        self,
        dataset,
        my_rank: int,
        num_nodes: int,
        batch_size: int
    ) -> DataLoader:
        """Split dataset across nodes by rank."""
        total = len(dataset)
        chunk = total // num_nodes
        start = my_rank * chunk
        end = start + chunk if my_rank < num_nodes - 1 else total
        subset = Subset(dataset, range(start, end))
        print(f"[ColonyML] Node {my_rank}: "
              f"samples {start}-{end} ({len(subset)} total)")
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True
        )

    def sync_gradients(
        self,
        model: nn.Module,
        nodes: List[dict],
        my_rank: int
    ):
        """Average gradients across all nodes via Ring AllReduce."""
        synced = 0
        for param in model.parameters():
            if param.grad is not None:
                try:
                    averaged = self.communicator.ring_allreduce(
                        param.grad,
                        nodes,
                        my_rank=my_rank
                    )
                    param.grad = averaged
                    synced += 1
                except Exception:
                    pass
        if synced > 0:
            print(f"[ColonyML] Synced {synced} gradient tensors")

    def train(
        self,
        model: nn.Module,
        dataset,
        epochs: int = 3,
        batch_size: int = 64,
        lr: float = 0.001,
        sync_every_n_batches: int = 10,
        on_epoch_end: Optional[Callable] = None
    ):
        """
        Main training loop — fully distributed, zero config.

        Args:
            model:    PyTorch model to train
            dataset:  Full training dataset (split across nodes)
            epochs:   Number of training epochs
            batch_size: Total batch size (split by scheduler)
            lr:       Learning rate
            sync_every_n_batches: How often to sync gradients
            on_epoch_end: Optional callback(epoch, loss, rank)
        """
        print(f"\n{'='*60}")
        print(f"ColonyML — Zero-Config Distributed Training")
        print(f"{'='*60}")

        # Step 1: Discover cluster
        nodes = self.discover_cluster()
        my_rank = self.get_my_rank(nodes)
        num_nodes = len(nodes)

        # Step 2: Assign batch sizes based on CPU power
        node_assignments = self.scheduler.assign_batch_sizes(
            nodes, batch_size
        )
        my_batch_size = node_assignments.get(
            self.local_ip, batch_size
        )

        print(f"\n[ColonyML] Rank {my_rank} | "
              f"Batch size: {my_batch_size} | "
              f"Nodes: {num_nodes}")

        # Step 3: Get data subset for this node
        loader = self.get_data_subset(
            dataset, my_rank, num_nodes, my_batch_size
        )

        # Step 4: Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Step 5: Barrier sync — all nodes start together
        self.barrier_sync(
            nodes, my_rank, barrier_name="training_start"
        )

        # Step 6: Training loop
        start_time = time.time()

        for epoch in range(epochs):
            print(f"\n[ColonyML] Epoch {epoch+1}/{epochs} "
                  f"| Rank {my_rank}")

            model.train()
            total_loss = 0.0
            batches = 0

            for batch_idx, (data, target) in enumerate(loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Sync gradients every N batches
                if (batch_idx + 1) % sync_every_n_batches == 0:
                    if num_nodes > 1:
                        self.sync_gradients(model, nodes, my_rank)

                optimizer.step()
                total_loss += loss.item()
                batches += 1

                if batch_idx % 100 == 0:
                    print(f"[ColonyML] Rank {my_rank} | "
                          f"Batch {batch_idx}/{len(loader)} | "
                          f"Loss: {loss.item():.4f}")

            avg_loss = total_loss / batches if batches > 0 else 0
            elapsed = time.time() - start_time

            print(f"[ColonyML] Epoch {epoch+1} complete | "
                  f"Avg loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

            if on_epoch_end:
                on_epoch_end(epoch, avg_loss, my_rank)

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"[ColonyML] Training complete!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Rank:       {my_rank}")
        print(f"Nodes:      {num_nodes}")
        print(f"{'='*60}")

        self.discovery.stop()
        return model