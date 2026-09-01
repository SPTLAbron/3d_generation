from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
EXPERIMENTS = OUTPUTS / "experiments"
RENDERS = OUTPUTS / "renders"
README_IMAGES = ROOT / "docs" / "images"
THRESHOLD = 0.5
VIEW = (22, -55)
LATENT_PROPERTIES = ("ball_radius", "support_sweep", "lower_base_radius")
LATENT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
VAE_SAMPLE_INDICES = (0, 4, 9, 14, 26, 44, 59, 81)
OPTIMIZATION_STEPS = (0, 5, 10, 15, 20, 25, 30)


def load_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    volume = np.squeeze(np.load(path))
    if volume.shape != (32, 32, 32):
        raise ValueError(f"Expected a 32x32x32 volume at {path}, got {volume.shape}")
    return volume >= THRESHOLD


def draw_voxels(ax, volume: np.ndarray, title: str = "") -> None:
    ax.voxels(volume, facecolors="#6f42c1", edgecolor="#3f276e", linewidth=0.08)
    ax.view_init(*VIEW)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=0)


def save_figure(fig, filename: str) -> None:
    README_IMAGES.mkdir(parents=True, exist_ok=True)
    path = README_IMAGES / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


def dataset_overview() -> None:
    voxel_dir = ROOT / "data" / "voxels"
    paths = sorted(voxel_dir.glob("*.npy"))
    if len(paths) < 12:
        raise FileNotFoundError(f"Need at least 12 dataset voxels in {voxel_dir}")
    indices = np.linspace(0, len(paths) - 1, 12, dtype=int)
    fig = plt.figure(figsize=(13, 9))
    for plot_index, source_index in enumerate(indices, start=1):
        ax = fig.add_subplot(3, 4, plot_index, projection="3d")
        draw_voxels(ax, load_volume(paths[source_index]))
    fig.suptitle("Procedurally Generated Trophy Dataset", fontsize=18)
    fig.tight_layout()
    save_figure(fig, "dataset_overview.png")


def latent_traversals() -> None:
    root = EXPERIMENTS / "latent_edits"
    fig = plt.figure(figsize=(15, 9))
    for row, parameter in enumerate(LATENT_PROPERTIES):
        for col, alpha in enumerate(LATENT_ALPHAS):
            path = root / parameter / f"alpha_{alpha:+.1f}_voxels.npy"
            ax = fig.add_subplot(
                len(LATENT_PROPERTIES), len(LATENT_ALPHAS),
                row * len(LATENT_ALPHAS) + col + 1, projection="3d"
            )
            draw_voxels(ax, load_volume(path), f"{alpha:+.0f}d" if row == 0 else "")
            if col == 0:
                ax.text2D(-0.18, 0.5, parameter.replace("_", " "),
                          transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=11)
    fig.suptitle("Semantic Latent Traversals", fontsize=18)
    fig.tight_layout()
    save_figure(fig, "latent_traversals.png")


def interpolation() -> None:
    root = EXPERIMENTS / "interpolation"
    paths = sorted(root.glob("*.npy"))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need interpolation arrays in {root}")
    selected = [paths[i] for i in np.linspace(0, len(paths) - 1, 7, dtype=int)]
    fig = plt.figure(figsize=(16, 3.5))
    for i, path in enumerate(selected):
        ax = fig.add_subplot(1, len(selected), i + 1, projection="3d")
        draw_voxels(ax, load_volume(path), f"t={i / (len(selected) - 1):.2f}")
    fig.suptitle("Autoencoder Latent Interpolation", fontsize=18)
    fig.tight_layout()
    save_figure(fig, "latent_interpolation.png")


def vae_samples() -> None:
    rows = (
        ("std = 1.0", EXPERIMENTS / "vae_samples"),
        ("std = 0.7", EXPERIMENTS / "vae_samples_std_0.7"),
        ("std = 0.5", EXPERIMENTS / "vae_samples_std_0.5"),
    )
    fig = plt.figure(figsize=(18, 7))
    for row, (label, directory) in enumerate(rows):
        for col, sample_index in enumerate(VAE_SAMPLE_INDICES):
            path = directory / f"sample_{sample_index:03d}.npy"
            ax = fig.add_subplot(len(rows), len(VAE_SAMPLE_INDICES),
                                  row * len(VAE_SAMPLE_INDICES) + col + 1,
                                  projection="3d")
            draw_voxels(ax, load_volume(path))
            if col == 0:
                ax.text2D(-0.15, 0.5, label, transform=ax.transAxes,
                          rotation=90, va="center", fontsize=11)
    fig.suptitle("VAE Samples at Different Latent Standard Deviations", fontsize=18)
    fig.tight_layout()
    save_figure(fig, "vae_samples.png")


def optimization() -> None:
    root = EXPERIMENTS / "shape_optimization"
    history_path = root / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    history = pd.read_csv(history_path)
    fig = plt.figure(figsize=(17, 7))
    grid = fig.add_gridspec(2, len(OPTIMIZATION_STEPS), height_ratios=(2.2, 1.0))
    for col, step in enumerate(OPTIMIZATION_STEPS):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        draw_voxels(ax, load_volume(root / f"step_{step:03d}.npy"), f"Step {step}")
    ax = fig.add_subplot(grid[1, :])
    ax.plot(history["step"], history["predicted_lower_base_radius"],
            label="Predicted lower-base radius", color="#6f42c1", linewidth=2)
    ax.plot(history["step"], history["predicted_ball_radius"],
            label="Predicted ball radius", color="#e07a35", linewidth=2)
    ax.set(xlabel="Optimization step", ylabel="Probe prediction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.suptitle("Gradient-Based Latent-Space Optimization", fontsize=18)
    fig.tight_layout()
    save_figure(fig, "latent_optimization.png")


def reconstruction() -> None:
    source = RENDERS / "ae_vs_vae_reconstruction.png"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run python src/analysis/compare_ae_vae.py first."
        )
    README_IMAGES.mkdir(parents=True, exist_ok=True)
    destination = README_IMAGES / "reconstructions.png"
    shutil.copy2(source, destination)
    print(f"Copied {destination.relative_to(ROOT)}")


FIGURES = {
    "dataset": dataset_overview,
    "reconstruction": reconstruction,
    "interpolation": interpolation,
    "traversals": latent_traversals,
    "samples": vae_samples,
    "optimization": optimization,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=FIGURES,
                        help="Generate only the named figures")
    parser.add_argument("--strict", action="store_true",
                        help="Stop at the first missing prerequisite")
    args = parser.parse_args()
    names = args.only or list(FIGURES)
    failures = []
    for name in names:
        try:
            FIGURES[name]()
        except (FileNotFoundError, ValueError, KeyError) as error:
            if args.strict:
                raise
            failures.append((name, error))
            print(f"Skipped {name}: {error}")
    if failures:
        print("\nSome figures were skipped because their inputs are missing.")
        print("Generate those experiment outputs, then run this command again.")


if __name__ == "__main__":
    main()
