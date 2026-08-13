"""
ColonyML — MNIST Training Example

This example shows how ColonyML coordinates distributed
training across multiple machines. Each machine trains
on a different subset of MNIST digits.

Single machine usage (simulation):
    python examples/train_mnist.py --simulate

Multi-machine usage:
    Machine 1: python examples/train_mnist.py --rank 0 --nodes 2
    Machine 2: python examples/train_mnist.py --rank 1 --nodes 2
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ------------------------------------------------
# Simple CNN for MNIST
# ------------------------------------------------

class MNISTNet(nn.Module):
    """
    Simple CNN for MNIST digit classification.
    Small enough to train quickly on CPU.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ------------------------------------------------
# Training functions
# ------------------------------------------------

def get_data_subset(rank: int, num_nodes: int, batch_size: int):
    """
    Get this node's subset of MNIST training data.
    Each node trains on a different portion.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Split dataset across nodes
    total = len(dataset)
    chunk_size = total // num_nodes
    start = rank * chunk_size
    end = start + chunk_size if rank < num_nodes - 1 else total

    subset = Subset(dataset, range(start, end))

    print(f"[Node {rank}] Training on samples {start} to {end} "
          f"({len(subset)} samples)")

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True
    )


def get_test_data(batch_size: int):
    """Get MNIST test data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    rank: int
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    batches = 0

    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # In real ColonyML distributed training:
        # gradients would be averaged here via Ring AllReduce
        # before optimizer.step()
        # comm.ring_allreduce(gradients, all_nodes, rank)

        optimizer.step()
        total_loss += loss.item()
        batches += 1

        if batch_idx % 50 == 0:
            print(f"[Node {rank}] Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return total_loss / batches


def evaluate(
    model: nn.Module,
    loader: DataLoader
) -> tuple:
    """Evaluate model. Returns (accuracy, avg_loss)."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)

    return accuracy, avg_loss


# ------------------------------------------------
# Simulate distributed training on one machine
# ------------------------------------------------

def simulate_distributed(epochs: int = 3, batch_size: int = 64):
    """
    Simulate ColonyML distributed training on one machine.
    Creates 2 virtual nodes, each training on half the data,
    then averages their gradients using Ring AllReduce simulation.
    """
    from colonyml.communicator import NodeCommunicator
    from colonyml.compressor import GradientCompressor, CompressionLevel
    from colonyml.scheduler import NodeScheduler

    print("=" * 60)
    print("ColonyML — Simulated Distributed MNIST Training")
    print("2 virtual nodes, Ring AllReduce gradient averaging")
    print("=" * 60)

    # Setup
    nodes = [
        {"ip": "node-0", "hostname": "virtual-node-0",
         "metrics": {"cores": 8, "cpu_usage": 20,
                     "cpu_freq_mhz": 3000,
                     "ram_available_gb": 8,
                     "ram_total_gb": 16}},
        {"ip": "node-1", "hostname": "virtual-node-1",
         "metrics": {"cores": 4, "cpu_usage": 30,
                     "cpu_freq_mhz": 2500,
                     "ram_available_gb": 4,
                     "ram_total_gb": 8}}
    ]

    scheduler = NodeScheduler()
    compressor = GradientCompressor()
    assignments = scheduler.assign_batch_sizes(nodes, batch_size)

    print(f"\nBatch assignment:")
    for node in nodes:
        ip = node["ip"]
        print(f"  {node['hostname']}: "
              f"{assignments[ip]} samples/batch")

    # Create two models (one per virtual node)
    model0 = MNISTNet()
    model1 = MNISTNet()

    # Share initial weights
    model1.load_state_dict(model0.state_dict())

    optimizer0 = optim.Adam(model0.parameters(), lr=0.001)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get data subsets
    loader0 = get_data_subset(0, 2, assignments["node-0"])
    loader1 = get_data_subset(1, 2, assignments["node-1"])
    test_loader = get_test_data(256)

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train both nodes
        loss0 = train_epoch(
            model0, loader0, optimizer0, criterion, rank=0
        )
        loss1 = train_epoch(
            model1, loader1, optimizer1, criterion, rank=1
        )

        # Average gradients via Ring AllReduce simulation
        print(f"\n[ColonyML] Averaging gradients via Ring AllReduce...")
        avg_count = 0
        for (name0, param0), (name1, param1) in zip(
            model0.named_parameters(),
            model1.named_parameters()
        ):
            if param0.grad is not None and param1.grad is not None:
                gradients = [param0.grad, param1.grad]
                averaged = NodeCommunicator.simulate_allreduce(
                    gradients
                )

                # Optional compression stats
                ratio = compressor.compression_ratio(
                    averaged, CompressionLevel.MEDIUM
                )
                avg_count += 1

                # Apply averaged gradient to both models
                param0.grad = averaged.clone()
                param1.grad = averaged.clone()

        print(f"[ColonyML] Averaged {avg_count} gradient tensors")
        print(f"[ColonyML] Compression ratio (Medium): {ratio:.1f}x")

        # Evaluate
        acc0, _ = evaluate(model0, test_loader)
        acc1, _ = evaluate(model1, test_loader)

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Node 0 accuracy: {acc0:.2f}%")
        print(f"  Node 1 accuracy: {acc1:.2f}%")
        print(f"  Avg loss node 0: {loss0:.4f}")
        print(f"  Avg loss node 1: {loss1:.4f}")
        print(f"  Time elapsed:    {elapsed:.1f}s")

    total_time = time.time() - start_time
    final_acc, _ = evaluate(model0, test_loader)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final accuracy:  {final_acc:.2f}%")
    print(f"Total time:      {total_time:.1f}s")
    print(f"Epochs:          {epochs}")
    print(f"Nodes simulated: 2")
    print("=" * 60)


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ColonyML MNIST Training Example"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate distributed training on one machine"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Total batch size across all nodes (default: 64)"
    )

    args = parser.parse_args()

    if args.simulate:
        simulate_distributed(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print("ColonyML MNIST Training Example")
        print("Usage:")
        print("  Simulate: python examples/train_mnist.py --simulate")
        print("  Epochs:   python examples/train_mnist.py "
              "--simulate --epochs 5")