"""Gráficos del laboratorio. Las tareas calculan; este módulo solo dibuja."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def use_lab_style() -> None:
    sns.set_theme(style="whitegrid")


def save_confusion(cm, best: str, path: str) -> None:
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Baja", "Sube"], yticklabels=["Baja", "Sube"])
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión\n{best}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_importances(imp: pd.Series, path: str, model_name: str = "Random Forest") -> None:
    plt.figure(figsize=(6, 4.5))
    imp.plot.barh(color="#2a7ab0")
    plt.title(f"Importancia de variables ({model_name}) – ¿sube/baja?")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_magnitude(index, real, pred, best: str, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(index, real * 100, lw=0.8, label="Real", alpha=0.8)
    ax[0].plot(index, pred * 100, lw=1.2, label=f"Predicho ({best})")
    ax[0].set_title("Magnitud del movimiento de mañana (%) – test")
    ax[0].legend()
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].scatter(real * 100, pred * 100, s=8, alpha=0.5)
    lim = max(real.max(), pred.max()) * 100
    ax[1].plot([0, lim], [0, lim], "r--", lw=1)
    ax[1].set_xlabel("Real (%)")
    ax[1].set_ylabel("Predicho (%)")
    ax[1].set_title("Real vs predicho")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_cluster_selection(ks, inertias, sils, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ks, inertias, "o-")
    ax[0].set_title("Codo (inercia)")
    ax[0].set_xlabel("k")
    ax[1].plot(ks, sils, "o-", color="darkorange")
    ax[1].set_title("Silhouette")
    ax[1].set_xlabel("k")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_cluster_pca(projected, labels, k_best: int, path: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=labels, palette="tab10", s=14, alpha=0.7)
    plt.title(f"Regímenes de mercado (K-Means, k={k_best}) – PCA 2D")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_cluster_price(dates, close, labels, path: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.scatter(dates, close, c=labels, cmap="tab10", s=6)
    plt.yscale("log")
    plt.title("Precio BTC (log) coloreado por régimen")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
