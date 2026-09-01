import csv
import random
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.trophy import generate_trophy

PARAMETER_RANGES = {
    "ball_radius": (0.78, 0.98),
    "ball_offset": (-0.92, -0.88),
    "support_sweep": (0.22, 0.34),
    "body_height": (2.65, 3.05),
    "body_bottom_radius": (0.48, 0.62),
    "body_top_radius": (0.78, 0.96),
    "lower_base_radius": (1.10, 1.40),
    "lower_base_height": (0.14, 0.22),
    "upper_base_radius": (0.90, 1.12),
    "upper_base_height": (0.12, 0.20),
}


def sample_parameters(rng):
    params = {name: rng.uniform(low, high) for name, (low, high) in PARAMETER_RANGES.items()}
    return params


def generate_dataset(num_samples=10000, seed=42, output_root=None):
    if output_root is None:
        output_root = PROJECT_ROOT / "data"

    output_root = Path(output_root)

    mesh_dir = output_root / "meshes"
    metadata_dir = output_root / "metadata"

    for old_mesh in mesh_dir.glob("trophy_*.obj"):
        old_mesh.unlink()

    mesh_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / "parameters.csv"

    rng = random.Random(seed)

    fieldnames = ["filename", *PARAMETER_RANGES.keys()]

    with metadata_path.open("w", newline="", encoding="utf-8",) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for index in range(num_samples):
            params = sample_parameters(rng)

            trophy = generate_trophy(**params)

            filename = f"trophy_{index:05d}.obj"

            output_path = mesh_dir / filename

            trophy.export(output_path)

            row = {"filename": filename}

            row.update({name: round(value, 6) for name, value in params.items()})

            writer.writerow(row)

            print(f"[{index + 1:03d}/{num_samples:03d}] " f"Saved {filename}")

    print()
    print("Dataset generation complete.")
    print(f"Meshes:   {mesh_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Samples:  {num_samples}")
    print(f"Seed:     {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
    )