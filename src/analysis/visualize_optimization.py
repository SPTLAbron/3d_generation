from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


INPUT_DIR = (
    ROOT
    / "outputs"
    / "experiments"
    / "shape_optimization"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "renders"
    / "shape_optimization"
)

THRESHOLD = 0.5

STEPS = [
    0,
    5,
    10,
    15,
    20,
    25,
    30,
]


def load_voxels(step):
    path = (
        INPUT_DIR
        / f"step_{step:03d}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing optimization output: {path}"
        )

    return np.load(path)


def render_trajectory():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(22, 4)
    )

    for i, step in enumerate(
        STEPS,
        start=1,
    ):
        probabilities = load_voxels(
            step
        )

        occupied = (
            probabilities
            >= THRESHOLD
        )

        ax = fig.add_subplot(
            1,
            len(STEPS),
            i,
            projection="3d",
        )

        ax.voxels(
            occupied
        )

        ax.set_title(
            f"Step {step}"
        )

        ax.set_axis_off()

        ax.set_box_aspect(
            (1, 1, 1)
        )

    fig.suptitle(
        "Latent-Space Shape Optimization",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "trajectory.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_parameter_history():
    history_path = (
        INPUT_DIR
        / "history.csv"
    )

    df = pd.read_csv(
        history_path
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["step"],
        df[
            "predicted_lower_base_radius"
        ],
        marker="o",
        markersize=3,
        label="Lower base radius",
    )

    ax.plot(
        df["step"],
        df[
            "predicted_ball_radius"
        ],
        marker="o",
        markersize=3,
        label="Ball radius",
    )

    ax.set_xlabel(
        "Optimization step"
    )

    ax.set_ylabel(
        "Probe-predicted parameter value"
    )

    ax.set_title(
        "Predicted Geometry During Optimization"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "parameter_history.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_latent_distance():
    history_path = (
        INPUT_DIR
        / "history.csv"
    )

    df = pd.read_csv(
        history_path
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["step"],
        df["latent_distance"],
        marker="o",
        markersize=3,
    )

    ax.set_xlabel(
        "Optimization step"
    )

    ax.set_ylabel(
        "Mean squared distance from z0"
    )

    ax.set_title(
        "Latent Distance During Optimization"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "latent_distance.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_bound_hits():
    history_path = (
        INPUT_DIR
        / "history.csv"
    )

    df = pd.read_csv(
        history_path
    )

    if (
        "latent_bound_hits"
        not in df.columns
    ):
        print(
            "No latent_bound_hits column. "
            "Skipping bound plot."
        )
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["step"],
        df["latent_bound_hits"],
        marker="o",
        markersize=3,
    )

    ax.set_xlabel(
        "Optimization step"
    )

    ax.set_ylabel(
        "Latent dimensions at ±3σ bound"
    )

    ax.set_title(
        "Latent Bound Saturation"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "latent_bound_hits.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def main():
    render_trajectory()

    plot_parameter_history()

    plot_latent_distance()

    plot_bound_hits()


if __name__ == "__main__":
    main()