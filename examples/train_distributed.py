"""
ColonyML — Real Distributed Training Across 2 Machines

Run on Machine 1 (Windows/rank 0):
    python examples/train_distributed.py --rank 0 --master-ip YOUR_IP

Run on Machine 2 (Mac/rank 1):
    python3 examples/train_distributed.py --rank 1 --master-ip YOUR_IP

Find your IP on Windows: ipconfig
Find your IP on Mac: ifconfig | grep inet
"""

import argparse
import socket
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from colonyml.communicator import NodeCommunicator
from colonyml.compressor import GradientCompressor, CompressionLevel
from colonyml.scheduler import NodeScheduler


class MNISTNet(nn.Module):
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


def get_data(rank, num_nodes, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        root="./data", train=True,
        download=True, transform=transform
    )
    total = len(dataset)
    chunk = total // num_nodes
    start = rank * chunk
    end = start + chunk if rank < num_nodes - 1 else total
    subset = Subset(dataset, range(start, end))
    print(f"[Node {rank}] Data: samples {start} to {end} "
          f"({len(subset)} total)")
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def get_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        root="./data", train=False,
        download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=256)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    return 100.0 * correct / total


def run_distributed(rank, num_nodes, master_ip, port, epochs, batch_size):
    local_ip = socket.gethostbyname(socket.gethostname())

    print(f"\n{'='*60}")
    print(f"ColonyML Distributed Training")
    print(f"Rank: {rank}/{num_nodes} | IP: {local_ip}")
    print(f"Master: {master_ip}:{port}")
    print(f"{'='*60}\n")

    # Define all nodes
    all_nodes = [
        {"ip": master_ip, "port": port,
         "hostname": "node-0", "cores": 14},
        {"ip": master_ip if rank == 1 else local_ip,
         "port": port, "hostname": "node-1", "cores": 8}
    ]

    # Fix node IPs based on rank
    if rank == 0:
        all_nodes[0]["ip"] = local_ip
    else:
        all_nodes[1]["ip"] = local_ip

    comm = NodeCommunicator(local_ip=local_ip, port=port)
    compressor = GradientCompressor()

    # Setup
    model = MNISTNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = get_data(rank, num_nodes, batch_size)
    test_loader = get_test_data()

    print(f"[Node {rank}] Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n[Node {rank}] Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        batches = 0

        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Average gradients via Ring AllReduce
            if batch_idx % 10 == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        try:
                            averaged = comm.ring_allreduce(
                                param.grad,
                                all_nodes,
                                my_rank=rank
                            )
                            param.grad = averaged
                        except Exception as e:
                            pass

            optimizer.step()
            total_loss += loss.item()
            batches += 1

            if batch_idx % 100 == 0:
                print(f"[Node {rank}] Batch {batch_idx} "
                      f"| Loss: {loss.item():.4f}")

        acc = evaluate(model, test_loader)
        elapsed = time.time() - start_time
        print(f"[Node {rank}] Epoch {epoch+1} done "
              f"| Accuracy: {acc:.2f}% "
              f"| Time: {elapsed:.1f}s")

    final_acc = evaluate(model, test_loader)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"[Node {rank}] Training Complete")
    print(f"Final accuracy: {final_acc:.2f}%")
    print(f"Total time:     {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True,
                        help="0 for Windows, 1 for Mac")
    parser.add_argument("--master-ip", type=str, required=True,
                        help="IP of the Windows machine")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--port", type=int, default=29500)
    args = parser.parse_args()

    run_distributed(
        rank=args.rank,
        num_nodes=2,
        master_ip=args.master_ip,
        port=args.port,
        epochs=args.epochs,
        batch_size=args.batch_size
    )