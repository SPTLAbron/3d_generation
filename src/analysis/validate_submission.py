from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (
    AE_CHECKPOINT,
    DATA_ROOT,
    EXPERIMENT_DIR,
    PARAM_COLUMNS,
    VAE_CHECKPOINT,
)
from src.data.dataset import TrophyDataset, get_datasets


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(condition, message):
    if condition:
        print(f"{PASS} {message}")
        return True

    print(f"{FAIL} {message}")
    return False


def warning(condition, message):
    if condition:
        print(f"{PASS} {message}")
    else:
        print(f"{WARN} {message}")


def main():
    print("=" * 72)
    print("3D TROPHY PROJECT - SUBMISSION VALIDATION")
    print("=" * 72)

    failures = 0

    print("\nDATASET")
    print("-" * 72)

    metadata_path = DATA_ROOT / "metadata" / "parameters.csv"
    voxel_dir = DATA_ROOT / "voxels"

    if not check(metadata_path.exists(), "Parameter metadata exists"):
        failures += 1
        df = None
    else:
        df = pd.read_csv(metadata_path)

    if df is not None:
        if not check(
            len(df) >= 10000,
            f"Dataset contains {len(df)} samples",
        ):
            failures += 1

        missing = [
            p for p in PARAM_COLUMNS
            if p not in df.columns
        ]

        if not check(
            len(missing) == 0,
            "All procedural parameters are present",
        ):
            failures += 1
            print("       Missing:", missing)

        if not check(
            "filename" in df.columns,
            "Filename column exists",
        ):
            failures += 1

    print("\nVOXELS")
    print("-" * 72)

    voxel_files = sorted(voxel_dir.glob("*.npy"))

    if not check(
        len(voxel_files) >= 10000,
        f"Found {len(voxel_files)} voxel files",
    ):
        failures += 1

    if voxel_files:
        sample = np.load(voxel_files[0])

        if not check(
            sample.shape == (32, 32, 32),
            f"Voxel resolution is {sample.shape}",
        ):
            failures += 1

        unique_values = np.unique(sample)

        warning(
            np.all(np.isin(unique_values, [0, 1])),
            "Voxel sample is binary occupancy data",
        )

    print("\nDATA SPLITS")
    print("-" * 72)

    try:
        train, val, test = get_datasets()

        total = len(train) + len(val) + len(test)

        check(
            total == len(TrophyDataset()),
            "Train/validation/test split covers full dataset",
        )

        print(
            f"       train={len(train)}, "
            f"validation={len(val)}, "
            f"test={len(test)}"
        )

    except Exception as exc:
        failures += 1
        print(f"{FAIL} Could not construct dataset splits")
        print(f"       {exc}")

    print("\nCHECKPOINTS")
    print("-" * 72)

    if not check(
        AE_CHECKPOINT.exists(),
        "AE checkpoint exists",
    ):
        failures += 1

    if not check(
        VAE_CHECKPOINT.exists(),
        "VAE checkpoint exists",
    ):
        failures += 1

    print("\nEXPERIMENTS")
    print("-" * 72)

    required_files = {
        "AE latent probe":
            EXPERIMENT_DIR / "latent_probe_results.csv",

        "Disentanglement raw results":
            EXPERIMENT_DIR / "disentanglement.csv",

        "Disentanglement matrix":
            EXPERIMENT_DIR / "disentanglement_matrix.csv",

        "Disentanglement ratios":
            EXPERIMENT_DIR / "disentanglement_ratios.csv",

        "Shape optimization history":
            EXPERIMENT_DIR / "shape_optimization" / "history.csv",

        "AE/VAE comparison":
            EXPERIMENT_DIR / "ae_vs_vae" / "comparison.json",

        "AE/VAE reconstruction metrics":
            EXPERIMENT_DIR / "ae_vs_vae" / "reconstruction_metrics.csv",

        "VAE standard-prior evaluation":
            EXPERIMENT_DIR / "vae_samples" / "summary.json",

        "VAE std=0.7 evaluation":
            EXPERIMENT_DIR / "vae_samples_std_0.7" / "summary.json",

        "VAE std=0.5 evaluation":
            EXPERIMENT_DIR / "vae_samples_std_0.5" / "summary.json",
    }

    for name, path in required_files.items():
        if not check(path.exists(), name):
            failures += 1
            print(f"       Expected: {path}")

    interpolation_dir = EXPERIMENT_DIR / "interpolation"
    interpolation_files = sorted(
        interpolation_dir.glob("*.npy")
    )

    if not check(
        len(interpolation_files) >= 30,
        f"AE interpolation contains "
        f"{len(interpolation_files)} steps",
    ):
        failures += 1

    latent_edit_dir = EXPERIMENT_DIR / "latent_edits"

    expected_edit_parameters = [
        "ball_radius",
        "support_sweep",
        "body_height",
        "body_bottom_radius",
        "body_top_radius",
        "lower_base_radius",
    ]

    for parameter in expected_edit_parameters:
        parameter_dir = latent_edit_dir / parameter

        if not check(
            parameter_dir.exists(),
            f"Latent edit: {parameter}",
        ):
            failures += 1

    print("\nFINAL SUMMARY")
    print("-" * 72)

    final_summary_dir = EXPERIMENT_DIR / "final_summary"

    warning(
        (final_summary_dir / "results_summary.json").exists(),
        "Combined results summary exists",
    )

    warning(
        (final_summary_dir / "results_overview.txt").exists(),
        "Human-readable results overview exists",
    )

    print("\nENVIRONMENT")
    print("-" * 72)

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\n" + "=" * 72)

    if failures == 0:
        print("SUBMISSION VALIDATION PASSED")
        print("All required project artifacts were found.")
    else:
        print(
            f"SUBMISSION VALIDATION FAILED: "
            f"{failures} required check(s) failed."
        )

    print("=" * 72)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()