from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INPUT_ROOT = ROOT / "outputs" / "experiments" / "latent_edits"
OUTPUT_ROOT = ROOT / "outputs" / "renders" / "latent_edits"

THRESHOLD = 0.5

ALPHAS = [
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
]


def plot_voxel(ax, probabilities, title):
    occupied = (
        probabilities
        >= THRESHOLD
    )

    ax.voxels(occupied)

    ax.set_title(title)

    ax.set_axis_off()

    ax.set_box_aspect((1, 1, 1))


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_dirs = sorted(
        p
        for p in INPUT_ROOT.iterdir()
        if p.is_dir()
    )

    for parameter_dir in parameter_dirs:
        parameter_name = (parameter_dir.name)

        fig = plt.figure(
            figsize=(18, 4)
        )

        for i, alpha in enumerate(ALPHAS, start=1):
            path = (
                parameter_dir
                / (
                    f"alpha_"
                    f"{alpha:+.1f}"
                    f"_voxels.npy"
                )
            )

            if not path.exists():
                continue

            probabilities = np.load(
                path
            )

            ax = fig.add_subplot(
                1,
                len(ALPHAS),
                i,
                projection="3d",
            )

            plot_voxel(
                ax,
                probabilities,
                f"alpha={alpha:+.1f}",
            )

        fig.suptitle(
            parameter_name,
            fontsize=16,
        )

        fig.tight_layout()

        output_path = (
            OUTPUT_ROOT
            / f"{parameter_name}.png"
        )

        fig.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved {output_path}"
        )


if __name__ == "__main__":
    main()