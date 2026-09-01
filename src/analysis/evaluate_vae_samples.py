from pathlib import Path
import json

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DIR = (
    ROOT
    / "outputs"
    / "experiments"
    / "vae_samples"
)

THRESHOLD = 1.0


def evaluate_sample(path):

    probabilities = np.load(path)

    occupied = (
        probabilities >= THRESHOLD
    )

    voxel_count = int(
        occupied.sum()
    )

    if voxel_count == 0:
        return {
            "voxel_count": 0,
            "components": 0,
            "largest_component_fraction": 0.0,
        }

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    labeled, num_components = ndimage.label(
        occupied,
        structure=structure,
    )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    component_sizes = component_sizes[1:]

    if len(component_sizes) == 0:
        largest_fraction = 0.0
    else:
        largest_component = int(
            component_sizes.max()
        )

        largest_fraction = (
            largest_component
            / voxel_count
        )

    return {
        "voxel_count": voxel_count,
        "components": int(num_components),
        "largest_component_fraction": float(
            largest_fraction
        ),
    }


def main():

    files = sorted(
        SAMPLE_DIR.glob(
            "sample_*.npy"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No VAE samples found in: "
            f"{SAMPLE_DIR}"
        )

    results = []

    for path in files:

        result = evaluate_sample(
            path
        )

        result["sample"] = path.name

        results.append(
            result
        )

        print(
            f"{path.name:<20} "
            f"voxels="
            f"{result['voxel_count']:5d} "
            f"components="
            f"{result['components']:3d} "
            f"largest="
            f"{result['largest_component_fraction']:.4f}"
        )

    largest_fractions = np.array(
        [
            result[
                "largest_component_fraction"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    component_counts = np.array(
        [
            result["components"]
            for result in results
        ],
        dtype=np.int64,
    )

    num_samples = len(results)

    mean_largest = float(
        largest_fractions.mean()
    )

    fully_connected_count = int(
        np.sum(
            component_counts == 1
        )
    )

    over_99_count = int(
        np.sum(
            largest_fractions > 0.99
        )
    )

    over_95_count = int(
        np.sum(
            largest_fractions > 0.95
        )
    )

    fully_connected_fraction = (
        fully_connected_count
        / num_samples
    )

    over_99_fraction = (
        over_99_count
        / num_samples
    )

    over_95_fraction = (
        over_95_count
        / num_samples
    )

    print()
    print(
        "------------------------------"
    )
    print(
        "VAE GENERATION SUMMARY"
    )
    print(
        "------------------------------"
    )

    print(
        f"Samples: {num_samples}"
    )

    print(
        "Mean largest-component fraction: "
        f"{mean_largest:.4f}"
    )

    print(
        "Fully connected: "
        f"{100 * fully_connected_fraction:.1f}%"
    )

    print(
        ">99% in largest component: "
        f"{100 * over_99_fraction:.1f}%"
    )

    print(
        ">95% in largest component: "
        f"{100 * over_95_fraction:.1f}%"
    )

    summary = {
        "sample_directory": SAMPLE_DIR.name,
        "threshold": THRESHOLD,
        "num_samples": num_samples,

        "mean_largest_component_fraction": (
            mean_largest
        ),

        "fully_connected_count": (
            fully_connected_count
        ),

        "fully_connected_fraction": (
            fully_connected_fraction
        ),

        "over_99_percent_count": (
            over_99_count
        ),

        "over_99_percent_fraction": (
            over_99_fraction
        ),

        "over_95_percent_count": (
            over_95_count
        ),

        "over_95_percent_fraction": (
            over_95_fraction
        ),
    }

    summary_path = (
        SAMPLE_DIR
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        f"Saved summary to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()