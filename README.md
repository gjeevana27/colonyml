# ColonyML

Zero-config distributed ML training for CPU clusters. Pool any laptops
into a training cluster with one command — no IP addresses, no config
files, no GPU required.

[![PyPI version](https://badge.fury.io/py/colonyml.svg)](https://pypi.org/project/colonyml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen.svg)](https://github.com/gjeevana27/colonyml)

```bash
pip install colonyml
colonyml join
```

---

## The Problem ColonyML Solves

Every existing distributed ML training tool requires manual setup:

```bash
# PyTorch torchrun — you must know IPs and set ranks manually:
torchrun \
  --nnodes=2 \
  --master_addr="192.168.1.45" \
  --master_port=29500 \
  --node_rank=0 \
  train.py
```

ColonyML removes all of it:

```bash
# Run the same command on every machine — nothing else needed:
colonyml train --script train.py
```

Machines find each other automatically. Ranks are assigned
automatically. Data is split automatically. Training starts
when everyone is ready.

---

## What Makes ColonyML Different

| | Most distributed training tools | ColonyML |
|---|---|---|
| Node discovery | Manual — you type IP addresses | Automatic via mDNS |
| Rank assignment | Manual — you set --node_rank | Automatic by IP order |
| GPU requirement | Required (CUDA) | Not required — CPU only |
| OS support | Usually Linux only | Windows, Mac, Linux |
| Authentication | Varies | HMAC-SHA256 built in |
| Single command train | No | Yes — colonyml train |

---

## Verified Results

Real distributed training across Windows + Mac on local WiFi:

```
Nodes:    2 (Windows 14-core Intel + Mac Apple M-series)
Dataset:  MNIST — 60,000 images split across nodes
Accuracy: 97.58%
Time:     347.9s
```

Simulated 2-node training on single machine:

```
Accuracy:    98.49%
Compression: 3.3x (Ethernet) / 33x (WiFi)
```

---

## Installation

```bash
pip install colonyml
```

Python 3.10+. Windows, Mac, Linux. No GPU needed.

---

## Quick Start

### Join a cluster

```bash
colonyml join
```

```
Cluster — 2 node(s)
Hostname        IP Address      Cores   Status
Jarvis          192.168.1.30    14      YOU
Pranav-MacBook  192.168.1.45    8       ready
```

### Train a model

Run the same command on every machine:

```bash
colonyml train --script examples/train_mnist.py --epochs 3 --wait 15
```

Your script must define a `train_colonyml` function:

```python
def train_colonyml(epochs, batch_size, wait, port):
    from colonyml.trainer import ColonyTrainer
    trainer = ColonyTrainer(port=port, wait=wait)
    trainer.train(model=model, dataset=dataset,
                  epochs=epochs, batch_size=batch_size)
```

---

## CLI Reference

| Command | Description |
|---|---|
| `colonyml join` | Announce this node and discover others |
| `colonyml train --script FILE` | Start distributed training |
| `colonyml status` | Show CPU and RAM metrics |
| `colonyml version` | Show version |

`colonyml train` options: `--epochs`, `--batch-size`, `--wait`, `--port`

---

## How It Works

**Auto-Discovery** — Each machine announces itself via mDNS
(the same protocol behind AirDrop). Other machines hear the
announcement and connect automatically. No IP addresses typed.

**Rank Assignment** — Nodes are sorted by IP. Lowest IP becomes
rank 0 (master). Every machine arrives at the same order
independently — no coordination message needed.

**Barrier Sync** — Rank 0 waits for READY from all nodes before
broadcasting GO. Nobody starts training until everyone is ready.

**Adaptive Scheduling** — ColonyML measures CPU cores, usage,
frequency, and RAM, then assigns proportionally larger batches
to stronger machines. All nodes finish each round together.

**Ring AllReduce** — After every N batches, gradients are averaged
by passing chunks around a ring. Communication load is spread
evenly — no single node gets overwhelmed.

**Gradient Compression** — Only the K largest gradient values are
sent over the network. On WiFi this achieves 33x compression
with minimal accuracy impact.

| Level  | Network       | Ratio |
|--------|---------------|-------|
| NONE   | >1000 Mbps    | 1x    |
| MEDIUM | 100-1000 Mbps | 3.3x  |
| HIGH   | <100 Mbps     | 33x   |

---

## Security

ColonyML uses HMAC-SHA256 authentication via `SecureCommunicator`.
Every gradient message is signed with a shared secret key — nodes
without the key cannot join the cluster or inject gradients.

```python
from colonyml.secure_communicator import SecureCommunicator

comm = SecureCommunicator(
    local_ip="192.168.1.30",
    port=29500,
    secret_key="your-shared-secret"
)
```

All nodes must use the same secret key. Invalid signatures are
rejected immediately.

---

## Architecture

```
colonyml/
├── discovery.py               mDNS auto-discovery
├── communicator.py            Ring AllReduce gradient sync
├── secure_communicator.py     HMAC-SHA256 authentication
├── compressor.py              TopK gradient compression
├── scheduler.py               CPU-aware batch scheduling
├── trainer.py                 Zero-config training coordinator
└── cli.py                     CLI entry point

tests/                         33 tests across 4 files
examples/train_mnist.py        MNIST distributed training demo
```

---

## Running Tests

```bash
git clone https://github.com/gjeevana27/colonyml.git
cd colonyml
pip install -e . && pip install pytest
pytest tests/ -v
# 33 passed
```

---

## Release History

- v0.1.0 — Initial PyPI release
- v0.1.2 — mDNS discovery, compression, CLI
- v0.1.3 — Ring AllReduce, 27 tests
- v0.1.5 — MNIST demo, 98.49% simulated accuracy
- v1.0.0 — ColonyTrainer, colonyml train command
- v1.0.1 — Real multi-machine verified (Windows + Mac, 97.58%)
- v1.0.2 — HMAC-SHA256 auth, SecureCommunicator, 33 tests

---

## Author

Jeevana Sai Gogineni


[GitHub](https://github.com/gjeevana27) · [LinkedIn](https://linkedin.com/in/jeevana-gogineni) · [PyPI](https://pypi.org/project/colonyml)

---

## License

MIT License