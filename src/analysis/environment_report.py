from pathlib import Path
import json
import platform
import sys

import matplotlib
import numpy
import pandas
import sklearn
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "experiments" / "environment.json"


def main():
    information = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "trimesh": trimesh.__version__,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "experiment": {
            "voxel_resolution": 32,
            "latent_dimension": 32,
            "random_seed": 42,
            "train_fraction": 0.80,
            "validation_fraction": 0.10,
            "test_fraction": 0.10,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(information, file, indent=2)

    print(json.dumps(information, indent=2))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()