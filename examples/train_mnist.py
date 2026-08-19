"""
ColonyML — MNIST Training Example

Single machine simulation:
    python examples/train_mnist.py --simulate --epochs 2

Zero-config distributed (run on EVERY machine):
    colonyml train --script examples/train_mnist.py --epochs 2 --wait 10
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ------------------------------------------------
# Model
# ------------------------------------------------

class MNISTNet(nn.Module):
    """Simple CNN for MNIST digit classification."""

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
# Data helpers
# ------------------------------------------------

def get_full_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


def get_data_subset(rank, num_nodes, batch_size):
    dataset = get_full_dataset()
    total = len(dataset)
    chunk = total // num_nodes
    start = rank * chunk
    end = start + chunk if rank < num_nodes - 1 else total
    subset = Subset(dataset, range(start, end))
    print(f"[Node {rank}] Samples {start} to {end} "
          f"({len(subset)} total)")
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def get_test_data(batch_size=256):
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


# ------------------------------------------------
# Training helpers
# ------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, rank):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
        if batch_idx % 100 == 0:
            print(f"[Node {rank}] Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")
    return total_loss / batches if batches > 0 else 0


def evaluate(model, loader):
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
# Simulation mode (single machine)
# ------------------------------------------------

def simulate_distributed(epochs=3, batch_size=64):
    """
    Simulate distributed training on one machine.
    Creates 2 virtual nodes with Ring AllReduce.
    """
    from colonyml.communicator import NodeCommunicator
    from colonyml.compressor import GradientCompressor, CompressionLevel
    from colonyml.scheduler import NodeScheduler

    print("=" * 60)
    print("ColonyML — Simulated Distributed MNIST Training")
    print("2 virtual nodes + Ring AllReduce gradient averaging")
    print("=" * 60)

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
        print(f"  {node['hostname']}: "
              f"{assignments[node['ip']]} samples/batch")

    model0 = MNISTNet()
    model1 = MNISTNet()
    model1.load_state_dict(model0.state_dict())

    optimizer0 = optim.Adam(model0.parameters(), lr=0.001)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loader0 = get_data_subset(0, 2, assignments["node-0"])
    loader1 = get_data_subset(1, 2, assignments["node-1"])
    test_loader = get_test_data(256)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        loss0 = train_epoch(
            model0, loader0, optimizer0, criterion, 0
        )
        loss1 = train_epoch(
            model1, loader1, optimizer1, criterion, 1
        )

        print(f"\n[ColonyML] Averaging gradients via Ring AllReduce...")
        avg_count = 0
        ratio = 1.0
        for (name0, param0), (name1, param1) in zip(
            model0.named_parameters(),
            model1.named_parameters()
        ):
            if param0.grad is not None and param1.grad is not None:
                gradients = [param0.grad, param1.grad]
                averaged = NodeCommunicator.simulate_allreduce(
                    gradients
                )
                ratio = compressor.compression_ratio(
                    averaged, CompressionLevel.MEDIUM
                )
                avg_count += 1
                param0.grad = averaged.clone()
                param1.grad = averaged.clone()

        print(f"[ColonyML] Averaged {avg_count} gradient tensors")
        print(f"[ColonyML] Compression ratio: {ratio:.1f}x")

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
# ColonyML distributed mode (called by colonyml train)
# ------------------------------------------------

def train_colonyml(
    epochs=3,
    batch_size=64,
    wait=10,
    port=29500
):
    """
    Called automatically by:
        colonyml train --script examples/train_mnist.py

    Run the SAME command on every machine.
    No --rank. No --master-ip. No config.
    """
    from colonyml.trainer import ColonyTrainer

    dataset = get_full_dataset()
    model = MNISTNet()

    trainer = ColonyTrainer(
        port=port,
        wait=wait        # ← fixed: was wait_seconds
    )

    trained_model = trainer.train(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size
    )

    print("\nEvaluating final model...")
    test_loader = get_test_data(256)
    acc, _ = evaluate(trained_model, test_loader)
    print(f"Final test accuracy: {acc:.2f}%")


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
        help="Total batch size (default: 64)"
    )

    args = parser.parse_args()

    if args.simulate:
        simulate_distributed(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print("ColonyML MNIST Training Example")
        print("\nUsage:")
        print("  Simulate (1 machine):")
        print("    python examples/train_mnist.py "
              "--simulate --epochs 2")
        print("\n  Distributed (run on every machine):")
        print("    colonyml train --script examples/train_mnist.py "
              "--epochs 2 --wait 10")