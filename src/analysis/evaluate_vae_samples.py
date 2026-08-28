from pathlib import Path

import numpy as np
from scipy.ndimage import label


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DIR = (
    ROOT
    / "outputs"
    / "experiments"
    / "vae_samples"
)

THRESHOLD = 0.5


def evaluate_sample(path):
    probabilities = np.load(path)

    voxels = probabilities > THRESHOLD

    occupied = int(voxels.sum())

    # 6-connected 3D components
    structure = np.zeros((3, 3, 3), dtype=int)

    structure[1, 1, 1] = 1
    structure[0, 1, 1] = 1
    structure[2, 1, 1] = 1
    structure[1, 0, 1] = 1
    structure[1, 2, 1] = 1
    structure[1, 1, 0] = 1
    structure[1, 1, 2] = 1

    labeled, num_components = label(
        voxels,
        structure=structure
    )

    if num_components == 0:
        largest_fraction = 0.0
    else:
        component_sizes = np.bincount(
            labeled.ravel()
        )[1:]

        largest_fraction = (
            component_sizes.max()
            / occupied
        )

    return (
        occupied,
        num_components,
        largest_fraction
    )


def main():
    paths = sorted(
        SAMPLE_DIR.glob("sample_*.npy")
    )

    results = []

    for path in paths:

        occupied, components, largest_fraction = (
            evaluate_sample(path)
        )

        results.append(
            largest_fraction
        )

        print(
            f"{path.name:20s} "
            f"voxels={occupied:5d} "
            f"components={components:3d} "
            f"largest={largest_fraction:.4f}"
        )

    results = np.array(results)

    print()
    print("------------------------------")
    print("VAE GENERATION SUMMARY")
    print("------------------------------")

    print(
        f"Samples: {len(results)}"
    )

    print(
        f"Mean largest-component fraction: "
        f"{results.mean():.4f}"
    )

    print(
        f"Fully connected: "
        f"{np.mean(results == 1.0) * 100:.1f}%"
    )

    print(
        f">99% in largest component: "
        f"{np.mean(results >= 0.99) * 100:.1f}%"
    )

    print(
        f">95% in largest component: "
        f"{np.mean(results >= 0.95) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()