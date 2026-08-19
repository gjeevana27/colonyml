# ColonyML

Zero-config distributed ML training for CPU clusters.
No IP addresses. No config files. No GPU required.

[![PyPI version](https://badge.fury.io/py/colonyml.svg)](https://pypi.org/project/colonyml/)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen.svg)](https://github.com/gjeevana27/colonyml)

```bash
pip install colonyml
colonyml join
```

---

## Verified Results

```
2 nodes: Windows + Mac on local WiFi
Dataset: MNIST 60,000 images
Accuracy: 97.58%
Compression: 33x over WiFi
Tests: 33 passing
```

---

## Install

```bash
pip install colonyml
```

Python 3.10+. Windows, Mac, Linux. No GPU needed.

---

## Quick Start

Run on every machine:

```bash
colonyml train --script train.py --epochs 3 --wait 15
```

Your script needs one function:

```python
def train_colonyml(epochs, batch_size, wait, port):
    from colonyml.trainer import ColonyTrainer
    trainer = ColonyTrainer(port=port, wait=wait)
    trainer.train(model=model, dataset=dataset,
                  epochs=epochs, batch_size=batch_size)
```

---

## CLI

| Command | Description |
|---|---|
| `colonyml join` | Join cluster, discover other nodes |
| `colonyml train --script FILE` | Start distributed training |
| `colonyml status` | Show CPU and RAM metrics |
| `colonyml version` | Show version |

---

## Features

- Auto-discovery via mDNS — no IP addresses needed
- Automatic rank assignment by IP order
- Adaptive batch scheduling by CPU power
- Ring AllReduce gradient synchronization
- TopK gradient compression — 33x over WiFi
- HMAC-SHA256 authentication

---

## Full Documentation

github.com/gjeevana27/colonyml

---

## Author

Jeevana Sai Gogineni


---

## License

MIT