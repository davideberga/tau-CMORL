# tau-cmorl

Implementation of the paper *"Multi-Constrained Reinforcement Learning under Full Signal
Temporal Logic Specifications"*

## Requirements

- Python 3.8.20

The experiments reported in the paper were conducted on a machine with the following specifications:
- OS: Linux Mint 22.1 (based on Ubuntu 24.04 LTS)
- CPU: AMD Ryzen 9 5950X 16-Core Processor
- RAM: 32 GB DDR4
- GPU: NVIDIA GeForce RTX 3090

## Installation

Create a python virtual environment and install the dependecies:

```bash
python3.8 -m venv .venv/
source .venv/bin/activate 
pip install -r requirements.txt
```
## Run

To run all the experiments use the following bash scripts:

```bash
source .venv/bin/activate 
./pendulum_experiments.sh
./rover_experiments.sh
```

