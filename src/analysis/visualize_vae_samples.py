from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DIRS = {
    "std = 1.0": (
        ROOT / "outputs" / "experiments" / "vae_samples"
    ),
    "std = 0.7": (
        ROOT / "outputs" / "experiments" / "vae_samples_std_0.7"
    ),
    "std = 0.5": (
        ROOT / "outputs" / "experiments" / "vae_samples_std_0.5"
    ),
}

OUTPUT_DIR = (
    ROOT / "outputs" / "renders"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "vae_sampling_comparison.png"
)

THRESHOLD = 0.5

SAMPLE_INDICES = [
    0,
    4,
    9,
    14,
    26,
    32,
    44,
    59,
    81,
    99,
]

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, directory in SAMPLE_DIRS.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing directory for {name}: "
                f"{directory}"
            )

    num_rows = len(SAMPLE_DIRS)
    num_cols = len(SAMPLE_INDICES)

    fig = plt.figure(
        figsize=(22, 8)
    )

    plot_index = 1

    for row_name, sample_dir in SAMPLE_DIRS.items():
        for sample_index in SAMPLE_INDICES:
            path = sample_dir / f"sample_{sample_index:03d}.npy"

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing sample: {path}"
                )

            probabilities = np.load(path)

            occupied = (
                probabilities >= THRESHOLD
            )

            ax = fig.add_subplot(
                num_rows,
                num_cols,
                plot_index,
                projection="3d",
            )

            ax.voxels(occupied)

            if row_name == "std = 1.0":
                ax.set_title(
                    f"Sample {sample_index}",
                    fontsize=9,
                )

            if sample_index == SAMPLE_INDICES[0]:
                ax.text2D(
                    -0.20,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=11,
                    va="center",
                )

            ax.set_axis_off()

            ax.set_box_aspect(
                (1, 1, 1)
            )

            plot_index += 1

    fig.suptitle(
        "VAE Sampling Variance Comparison",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()