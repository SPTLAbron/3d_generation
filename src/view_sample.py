from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = (
    ROOT
    / "outputs"
    / "experiments"
    / "latent_edit_ball_radius"
)

samples = [
    "alpha_-6.npy",
    "alpha_-3.npy",
    "alpha_+0.npy",
    "alpha_+3.npy",
    "alpha_+6.npy",
]

for filename in samples:

    path = SAMPLE_DIR / filename

    voxels = np.load(path)

    print(
        filename,
        "min=", voxels.min(),
        "max=", voxels.max(),
        "mean=", voxels.mean(),
    )

    voxels = voxels > 0.5

    fig = plt.figure()

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.voxels(voxels)

    ax.set_title(filename)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()