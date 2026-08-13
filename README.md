# ColonyML

> Zero-config distributed ML training for CPU clusters.

[![PyPI version](https://badge.fury.io/py/colonyml.svg)](https://pypi.org/project/colonyml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pool any laptops into a training cluster — no IP addresses,
no config files, no GPU required. Just:

```bash
pip install colonyml
colonyml join
```

---

## The Problem

Training ML models on one laptop is slow. You might have
multiple machines sitting idle. Existing distributed training
tools require manual IP configuration, Linux, or expensive GPUs.

ColonyML fixes this. Any laptop. Any OS. Zero setup.

---

## How It Works

```
Laptop 1 → colonyml join   (announces itself on network)
Laptop 2 → colonyml join   (finds Laptop 1 automatically)

Both laptops train together:
- Auto-discovery via mDNS (like AirDrop for ML)
- Ring AllReduce for gradient synchronization
- Adaptive batch sizing based on CPU power
- Gradient compression over WiFi (up to 50x)
```

---

## Features

- Zero config — no IP addresses, no config files
- Auto-discovery — machines find each other via mDNS
- Ring AllReduce — efficient gradient averaging across nodes
- Adaptive scheduling — faster machines get bigger batches
- Gradient compression — up to 50x smaller over WiFi
- Any hardware — works on any CPU laptop, any OS

---

## Installation

```bash
pip install colonyml
```

---

## Status


Active Development — v0.1.3 released.

- mDNS auto-discovery working
- Gradient compression up to 33x
- Ring AllReduce implemented and tested
- 27 tests passing
- CLI: colonyml join / status / version

Star the repo to follow progress.

---


## Roadmap

- v0.1.0 — PyPI release, project structure
- v0.1.1 — Package name finalized, setup complete
- v0.1.2 — mDNS auto-discovery, gradient compression,
            adaptive CPU scheduler, CLI (colonyml join / status)
- v0.1.3 — Ring AllReduce gradient synchronization, 27 tests passing
- v0.1.4 — MNIST training example across 2 machines (coming soon)
- v1.0.0 — stable release, full documentation

---

## Author

Jeevana Sai Gogineni
MS Data Science · University of Maryland

GitHub: https://github.com/gjeevana27
LinkedIn: https://linkedin.com/in/jeevana-gogineni

---

## License

MIT License — free to use, modify, and distribute.