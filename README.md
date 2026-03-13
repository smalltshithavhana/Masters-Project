CASTLE Experiment 1 (Graph-Discovery Priority) — Reproducible Runner

What this project does
- Runs Experiment 1 as a controlled sweep over:
  * number of nodes
  * average graph degree (via edges = int(nodes * degree_factor))
  * dataset size
- Trains a modified CASTLE model with:
  * acyclicity enforcement via augmented Lagrangian (rho/alpha updates)
  * SHD-based model selection during training (select best adjacency + threshold)
  * structured logging (text + JSONL + CSV)
- Produces the required plots (saved as PNG) and all raw arrays (saved as NPY).

Quick start
1) Create environment (example):
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2) Run the full sweep:
   python run_experiment1.py --outputs Outputs

Outputs
- Outputs/<run_id>/
  - logs/
    - run.log                (full training + experiment log text)
    - runs_summary.csv        (one row per (nodes, degree, dataset_size, seed))
  - plots/
    - nodes_vs_kl.png
    - h_vs_nodes_degree.png
    - shd_vs_nodes_degree.png
    - thr_vs_nodes_degree.png
    - dataset_size_vs_shd__nodes_*.png
    - dataset_size_vs_kl__nodes_*.png
  - runs/
    - one folder per configuration with:
      * training.jsonl
      * summary.json
      * adj_true.npy
      * adj_learned_best.npy

Notes
- This is designed for Experiment 1 (structure recovery). It sets supervised_loss_weight=0 by default.
- KL here is computed over adjacency distributions (structural proxy), not full joint P(X) KL.


## GPU / Cluster (NVIDIA) setup

This project is compatible with NVIDIA GPUs on Linux clusters.

Recommended install (official TensorFlow pip guidance):
- Install GPU build via:
  pip install tensorflow[and-cuda]
- Ensure the node has a recent NVIDIA driver and compatible CUDA/cuDNN.

TensorFlow's official pip instructions list:
- pip install tensorflow[and-cuda]
- NVIDIA driver (Linux) >= 525.60.13
- CUDA Toolkit 12.3
- cuDNN 8.9.7

See TensorFlow official install docs for the current compatibility details.

### SLURM
Use:
  sbatch slurm_run_experiment1.sh

### Notes on TF1-style sessions
This project uses tf.compat.v1 sessions (to stay aligned with your CASTLE code).
We enable GPU memory growth to avoid grabbing all GPU memory at startup.
