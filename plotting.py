\
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_fig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_nodes_vs_kl(df: pd.DataFrame, out_path: Path) -> None:
    # aggregate over degree_factor and dataset_size
    g = df.groupby("nodes", as_index=False)["kl_true_learned"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(g["nodes"], g["kl_true_learned"], marker="o")
    plt.xlabel("Total number of nodes")
    plt.ylabel("KL divergence (true -> learned) over adjacency distribution")
    plt.title("Nodes vs KL (structural proxy)")
    save_fig(out_path)


def plot_h_vs_nodes_degree(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for deg in sorted(df["degree_factor"].unique()):
        sub = df[df["degree_factor"] == deg].groupby("nodes", as_index=False)["h_value_end"].mean()
        plt.plot(sub["nodes"], sub["h_value_end"], marker="o", label=f"degree_factor={deg:g}")
    plt.xlabel("Node size")
    plt.ylabel("h(W) end-of-training")
    plt.title("h-value vs node size (by graph degree)")
    plt.legend()
    save_fig(out_path)


def plot_shd_vs_nodes_degree(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for deg in sorted(df["degree_factor"].unique()):
        sub = df[df["degree_factor"] == deg].groupby("nodes", as_index=False)["best_shd"].mean()
        plt.plot(sub["nodes"], sub["best_shd"], marker="o", label=f"degree_factor={deg:g}")
    plt.xlabel("Node size")
    plt.ylabel("Best SHD (lower is better)")
    plt.title("SHD vs node size (by graph degree)")
    plt.legend()
    save_fig(out_path)


def plot_thr_vs_nodes_degree(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for deg in sorted(df["degree_factor"].unique()):
        sub = df[df["degree_factor"] == deg].groupby("nodes", as_index=False)["best_thr"].mean()
        plt.plot(sub["nodes"], sub["best_thr"], marker="o", label=f"degree_factor={deg:g}")
    plt.xlabel("Node size")
    plt.ylabel("Selected reconstruction threshold (best by SHD)")
    plt.title("Threshold vs node size (by graph degree)")
    plt.legend()
    save_fig(out_path)


def plot_dataset_size_slices(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_list = sorted(df["nodes"].unique())
    for n in nodes_list:
        subn = df[df["nodes"] == n]
        # SHD vs dataset size by degree
        plt.figure(figsize=(9, 6))
        for deg in sorted(subn["degree_factor"].unique()):
            sub = subn[subn["degree_factor"] == deg].groupby("dataset_size", as_index=False)["best_shd"].mean()
            plt.plot(sub["dataset_size"], sub["best_shd"], marker="o", label=f"degree_factor={deg:g}")
        plt.xlabel("Dataset size (N)")
        plt.ylabel("Best SHD")
        plt.title(f"Dataset size vs SHD (nodes={n})")
        plt.legend()
        save_fig(out_dir / f"dataset_size_vs_shd__nodes_{n}.png")

        # KL vs dataset size by degree
        plt.figure(figsize=(9, 6))
        for deg in sorted(subn["degree_factor"].unique()):
            sub = subn[subn["degree_factor"] == deg].groupby("dataset_size", as_index=False)["kl_true_learned"].mean()
            plt.plot(sub["dataset_size"], sub["kl_true_learned"], marker="o", label=f"degree_factor={deg:g}")
        plt.xlabel("Dataset size (N)")
        plt.ylabel("KL divergence (true -> learned)")
        plt.title(f"Dataset size vs KL (nodes={n})")
        plt.legend()
        save_fig(out_dir / f"dataset_size_vs_kl__nodes_{n}.png")
