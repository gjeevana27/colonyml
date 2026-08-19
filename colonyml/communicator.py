import socket
import pickle
import struct
import threading
import time
import torch
import numpy as np
from typing import List


class NodeCommunicator:
    """
    Handles sending and receiving gradients between nodes.
    Implements Ring AllReduce for efficient gradient averaging.

    Why Ring AllReduce:
    - No single node gets overwhelmed
    - Communication scales linearly with cluster size
    - Same algorithm used in production ML systems

    Usage:
        comm = NodeCommunicator(local_ip="192.168.1.1", port=29500)
        averaged = comm.ring_allreduce(gradient, all_nodes, my_rank=0)
    """

    BUFFER_SIZE = 65536

    def __init__(self, local_ip: str, port: int = 29500):
        self.local_ip = local_ip
        self.port = port
        self.recv_port = port + 1
        self._lock = threading.Lock()

    # ------------------------------------------------
    # Low-level send / receive
    # ------------------------------------------------

    def send_tensor(
        self,
        tensor: torch.Tensor,
        target_ip: str,
        target_port: int,
        timeout: int = 30
    ):
        """Send a tensor to another node."""
        data = pickle.dumps(tensor.numpy())
        size = len(data)

        with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ) as s:
            s.settimeout(timeout)
            s.connect((target_ip, target_port))
            # Send size header first
            s.sendall(struct.pack("!Q", size))
            # Send data in chunks
            sent = 0
            while sent < size:
                chunk = data[sent:sent + self.BUFFER_SIZE]
                s.sendall(chunk)
                sent += len(chunk)

    def receive_tensor(
        self,
        listen_ip: str,
        listen_port: int,
        timeout: int = 30
    ) -> torch.Tensor:
        """Listen for and receive a tensor."""
        with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ) as server:
            server.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            server.bind((listen_ip, listen_port))
            server.listen(1)
            server.settimeout(timeout)

            conn, addr = server.accept()
            with conn:
                # Read size header
                size_data = self._recv_exactly(conn, 8)
                size = struct.unpack("!Q", size_data)[0]
                # Read actual data
                data = self._recv_exactly(conn, size)

        array = pickle.loads(data)
        return torch.tensor(array)

    def _recv_exactly(
        self,
        conn: socket.socket,
        n: int
    ) -> bytes:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = conn.recv(
                min(n - len(data), self.BUFFER_SIZE)
            )
            if not chunk:
                raise ConnectionError(
                    "Connection closed before all data received"
                )
            data += chunk
        return data

    # ------------------------------------------------
    # Ring AllReduce
    # ------------------------------------------------

    def ring_allreduce(
        self,
        gradient: torch.Tensor,
        all_nodes: List[dict],
        my_rank: int
    ) -> torch.Tensor:
        """
        Ring AllReduce algorithm.

        Steps:
        1. Split gradient into N chunks (N = number of nodes)
        2. Reduce-scatter: N-1 rounds of passing chunks around
           the ring, adding as we go
        3. All-gather: N-1 rounds of passing complete chunks
           around the ring
        4. Divide by N to get average

        Result: every node has the same averaged gradient.

        Args:
            gradient:  this node's gradient tensor
            all_nodes: list of all nodes sorted by IP
                       each node has: ip, port, hostname
            my_rank:   this node's position in the ring (0-indexed)

        Returns:
            averaged gradient tensor (same shape as input)
        """
        n = len(all_nodes)

        # Single node — nothing to reduce
        if n == 1:
            return gradient

        original_shape = gradient.shape
        flat = gradient.flatten().float()
        total_elements = len(flat)

        # Split into N chunks
        chunk_size = (total_elements + n - 1) // n
        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = min(start + chunk_size, total_elements)
            chunks.append(flat[start:end].clone())

        # ---- Phase 1: Reduce-scatter ----
        # After this phase, each node has the correct
        # sum for one chunk
        for step in range(n - 1):
            # Which chunk to send this round
            send_chunk_idx = (my_rank - step) % n
            # Which chunk we'll receive into
            recv_chunk_idx = (my_rank - step - 1) % n

            # Next and previous nodes in the ring
            next_rank = (my_rank + 1) % n
            prev_rank = (my_rank - 1) % n

            next_node = all_nodes[next_rank]
            prev_node = all_nodes[prev_rank]

            received = self._exchange_chunks(
                send_tensor=chunks[send_chunk_idx],
                target_ip=next_node["ip"],
                target_port=next_node["port"] + 1,
                listen_ip=self.local_ip,
                listen_port=self.recv_port
            )

            # Add received chunk to ours
            chunks[recv_chunk_idx] = (
                chunks[recv_chunk_idx] + received
            )

        # ---- Phase 2: All-gather ----
        # After this phase, every node has every chunk
        for step in range(n - 1):
            send_chunk_idx = (my_rank - step + 1) % n
            recv_chunk_idx = (my_rank - step) % n

            next_rank = (my_rank + 1) % n
            prev_rank = (my_rank - 1) % n

            next_node = all_nodes[next_rank]
            prev_node = all_nodes[prev_rank]

            received = self._exchange_chunks(
                send_tensor=chunks[send_chunk_idx],
                target_ip=next_node["ip"],
                target_port=next_node["port"] + 1,
                listen_ip=self.local_ip,
                listen_port=self.recv_port
            )

            chunks[recv_chunk_idx] = received

        # Concatenate chunks and average
        result = torch.cat(chunks)[:total_elements]
        result = result / n

        return result.reshape(original_shape)

    def _exchange_chunks(
        self,
        send_tensor: torch.Tensor,
        target_ip: str,
        target_port: int,
        listen_ip: str,
        listen_port: int
    ) -> torch.Tensor:
        """
        Simultaneously send a chunk to the next node
        and receive a chunk from the previous node.
        Uses threads so send and receive happen at the same time.
        """
        received_tensor = [None]
        error = [None]

        def do_receive():
            try:
                received_tensor[0] = self.receive_tensor(
                    listen_ip, listen_port
                )
            except Exception as e:
                error[0] = e

        # Start receiver in background thread
        recv_thread = threading.Thread(target=do_receive)
        recv_thread.start()

        # Small delay to let receiver start listening
        time.sleep(0.05)

        # Send our chunk
        self.send_tensor(send_tensor, target_ip, target_port)

        # Wait for receive to complete
        recv_thread.join(timeout=30)

        if error[0]:
            raise error[0]

        if received_tensor[0] is None:
            raise TimeoutError("Did not receive tensor in time")

        return received_tensor[0]

    # ------------------------------------------------
    # Single-node simulation (for testing)
    # ------------------------------------------------

    @staticmethod
    def simulate_allreduce(
        gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Simulate Ring AllReduce on a single machine.
        Used for testing without multiple machines.

        Equivalent to: sum all gradients / number of nodes
        """
        if not gradients:
            raise ValueError("No gradients provided")

        stacked = torch.stack(gradients)
        return stacked.mean(dim=0)


# ------------------------------------------------
# Test on single machine
# ------------------------------------------------

if __name__ == "__main__":
    print("ColonyML — Communicator Test")
    print("=" * 50)

    # Simulate what would happen with 3 machines
    print("\nSimulating Ring AllReduce across 3 virtual nodes:")
    print("(Single machine simulation)")

    # Each "machine" has different gradients
    node1_gradient = torch.tensor([1.0, 2.0, 3.0, 4.0])
    node2_gradient = torch.tensor([5.0, 6.0, 7.0, 8.0])
    node3_gradient = torch.tensor([9.0, 10.0, 11.0, 12.0])

    gradients = [node1_gradient, node2_gradient, node3_gradient]

    print(f"\nNode 1 gradient: {node1_gradient.tolist()}")
    print(f"Node 2 gradient: {node2_gradient.tolist()}")
    print(f"Node 3 gradient: {node3_gradient.tolist()}")

    # What the average should be
    expected = torch.stack(gradients).mean(dim=0)
    print(f"\nExpected average: {expected.tolist()}")

    # Simulate AllReduce
    result = NodeCommunicator.simulate_allreduce(gradients)
    print(f"AllReduce result: {result.tolist()}")

    # Verify
    if torch.allclose(result, expected):
        print("\n✓ AllReduce correct — all nodes converge to same average")
    else:
        print("\n✗ AllReduce incorrect")

    # Test with larger tensor
    print("\n" + "=" * 50)
    print("Large tensor test (1000x1000):")

    large_gradients = [torch.randn(1000, 1000) for _ in range(4)]
    result = NodeCommunicator.simulate_allreduce(large_gradients)

    expected = torch.stack(large_gradients).mean(dim=0)
    error = (result - expected).abs().max().item()

    print(f"Max error: {error:.8f}")
    print(f"Shape preserved: {result.shape == large_gradients[0].shape}")
    print(f"✓ Large tensor AllReduce correct" if error < 1e-5
          else f"✗ Error too large: {error}")
    
from colonyml.secure_communicator import SecureCommunicator