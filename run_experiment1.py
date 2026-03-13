\
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

import os
try:
    import tensorflow as _tfgpu
    _g = _tfgpu.config.list_physical_devices('GPU')
    # Avoid TF grabbing all memory
    for _dev in _g:
        try:
            _tfgpu.config.experimental.set_memory_growth(_dev, True)
        except Exception:
            pass
except Exception:
    pass

from data_gen import random_dag, gen_data_nonlinear
from metrics import build_acyclic_graph_from_weights, compute_kl_adj_distributions, compute_shd_from_weights, adjacency_from_nx
from plotting import (
    plot_nodes_vs_kl,
    plot_h_vs_nodes_degree,
    plot_shd_vs_nodes_degree,
    plot_thr_vs_nodes_degree,
    plot_dataset_size_slices,
)
from castle_mod import CASTLE


def make_run_dir(outputs_root: Path, run_id: Optional[str] = None) -> Path:
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_root / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("exp1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def split_data(X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n = X.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = X[idx[:n_train]]
    val = X[idx[n_train:n_train + n_val]]
    test = X[idx[n_train + n_val:]]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str, default="Outputs")
    parser.add_argument("--run_id", type=str, default=None)

    # sweep controls
    parser.add_argument("--nodes_list", type=str, default="5,10,15,20")
    parser.add_argument("--degree_factors", type=str, default="1,2,3")
    parser.add_argument("--dataset_sizes", type=str, default="500,1000,5000")
    parser.add_argument("--seeds", type=str, default="42,43")

    # generator
    parser.add_argument("--sem_type", type=str, default="square", choices=["square", "sigmoid"])
    parser.add_argument("--noise_scale", type=float, default=1.0)

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_hidden", type=int, default=32)
    parser.add_argument("--subset_nodes_factor", type=float, default=0.6, help="subset_nodes = max(1, int(nodes*factor))")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--DAG_min", type=float, default=0.5)

    # selection
    parser.add_argument("--thr_grid", type=str, default="0.05,0.1,0.15,0.2,0.3")

    args = parser.parse_args()

    run_dir = make_run_dir(Path(args.outputs), args.run_id)
    logger = setup_logger(run_dir / "logs" / "run.log")

    nodes_list = [int(x.strip()) for x in args.nodes_list.split(",") if x.strip()]
    degree_factors = [float(x.strip()) for x in args.degree_factors.split(",") if x.strip()]
    dataset_sizes = [int(x.strip()) for x in args.dataset_sizes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    thr_grid = [float(x.strip()) for x in args.thr_grid.split(",") if x.strip()]

    logger.info("=== Experiment 1 sweep starting ===")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"nodes_list={nodes_list}")
    logger.info(f"degree_factors={degree_factors}")
    logger.info(f"dataset_sizes={dataset_sizes}")
    logger.info(f"seeds={seeds}")
    logger.info(f"sem_type={args.sem_type}, noise_scale={args.noise_scale}")

    rows: List[Dict] = []

    for nodes in nodes_list:
        node_names = [str(i) for i in range(nodes)]
        for degree_factor in degree_factors:
            edges = int(max(nodes - 1, round(nodes * degree_factor)))
            avg_degree = edges / nodes
            for dataset_size in dataset_sizes:
                for seed in seeds:
                    cfg_id = f"nodes{nodes}_deg{degree_factor:g}_N{dataset_size}_seed{seed}"
                    cfg_dir = run_dir / "runs" / cfg_id
                    cfg_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(f"--- Running {cfg_id} ---")

                    # 1) Generate ground-truth DAG + data
                    G_true = random_dag(nodes=nodes, edges=edges, seed=seed)
                    X = gen_data_nonlinear(G_true, n=dataset_size, sem_type=args.sem_type, noise_scale=args.noise_scale, seed=seed)

                    # standardize (helps optimization stability)
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)

                    X_train, X_val, X_test = split_data(Xs, seed=seed)

                    # save true adjacency
                    adj_true = adjacency_from_nx(G_true, node_names)
                    np.save(cfg_dir / "adj_true.npy", adj_true)

                    # 2) Train CASTLE
                    subset_nodes = max(1, int(nodes * args.subset_nodes_factor))

                    model = CASTLE(
                        num_train=X_train.shape[0],
                        lr=args.lr,
                        batch_size=args.batch_size,
                        num_inputs=nodes,
                        num_outputs=1,
                        n_hidden=args.n_hidden,
                        reg_beta=5.0,
                        DAG_min=args.DAG_min,
                        reconstruction_loss_weight=10.0,
                        dag_penalty_weight=25.0,
                        supervised_loss_weight=0.0,
                        max_steps=args.max_steps,
                        patience=args.patience,
                        ckpt_file=str(cfg_dir / "tmp.ckpt"),
                        seed=seed,
                    )

                    fit_res = model.fit(
                        X_train=X_train,
                        X_val=X_val,
                        subset_nodes=subset_nodes,
                        adj_true=adj_true,
                        shd_threshold_grid=thr_grid,
                        logger=logger,
                        training_jsonl_path=str(cfg_dir / "training.jsonl"),
                        seed=seed,
                    )

                    # 3) Build DAG from best_W using best_thr and compute metrics
                    W_best = fit_res.best_W
                    np.save(cfg_dir / "adj_learned_best.npy", W_best)

                    G_learned = build_acyclic_graph_from_weights(W_best, node_names, thr=fit_res.best_thr)

                    shd_info = compute_shd_from_weights(G_true, W_best, node_names, thr=fit_res.best_thr)
                    kl_info = compute_kl_adj_distributions(G_true, G_learned, node_names)

                    summary = {
                        "cfg_id": cfg_id,
                        "nodes": nodes,
                        "edges": edges,
                        "degree_factor": degree_factor,
                        "avg_degree": avg_degree,
                        "dataset_size": dataset_size,
                        "seed": seed,
                        "subset_nodes": subset_nodes,
                        "best_shd": shd_info["shd"],
                        "best_thr": fit_res.best_thr,
                        "h_value_end": fit_res.h_value_end,
                        **kl_info,
                        **{f"shd_{k}": v for k, v in shd_info.items()},
                    }

                    (cfg_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    rows.append(summary)

                    logger.info(f"Completed {cfg_id} | SHD={summary['best_shd']} | KL={summary['kl_true_learned']:.4f} | h_end={summary['h_value_end']:.3e} | thr={summary['best_thr']:.3f}")

    # write global summary
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "logs" / "runs_summary.csv", index=False)

    # produce required plots
    plots_dir = run_dir / "plots"
    plot_nodes_vs_kl(df, plots_dir / "nodes_vs_kl.png")
    plot_h_vs_nodes_degree(df, plots_dir / "h_vs_nodes_degree.png")
    plot_shd_vs_nodes_degree(df, plots_dir / "shd_vs_nodes_degree.png")
    plot_thr_vs_nodes_degree(df, plots_dir / "thr_vs_nodes_degree.png")
    plot_dataset_size_slices(df, plots_dir)

    logger.info("=== Experiment 1 sweep finished ===")
    logger.info(f"Summary CSV: {run_dir / 'logs' / 'runs_summary.csv'}")
    logger.info(f"Plots in: {plots_dir}")


if __name__ == "__main__":
    main()
