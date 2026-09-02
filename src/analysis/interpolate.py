from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import get_loaders
from src.models.autoencoder import Autoencoder3D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FRAMES = 60
FPS = 20
THRESHOLD = 0.5


def render_animation(volumes, output_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index):
        ax.clear()

        volume = volumes[frame_index] >= THRESHOLD
        t = frame_index / (len(volumes) - 1)

        ax.voxels(
            volume,
            facecolors="#6f42c1",
            edgecolor="#3f276e",
            linewidth=0.08,
        )

        ax.view_init(elev=22, azim=-55)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)
        ax.set_axis_off()
        ax.set_title(f"Latent interpolation — t={t:.2f}")

    animation = FuncAnimation(
        fig,
        update,
        frames=len(volumes),
        interval=1000 / FPS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output_path,
        writer=PillowWriter(fps=FPS),
        dpi=120,
    )

    plt.close(fig)
    print(f"Saved animation: {output_path}")


def main():
    _, _, _, _, _, test_dataset = get_loaders(batch_size=1)

    model = Autoencoder3D(32).to(DEVICE)
    model.load_state_dict(
        torch.load(
            ROOT / "outputs" / "checkpoints" / "ae_best.pt",
            map_location=DEVICE,
        )
    )
    model.eval()

    a = test_dataset[0][0].unsqueeze(0).to(DEVICE)
    b = test_dataset[len(test_dataset) // 2][0].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        za = model.encode(a)
        zb = model.encode(b)

        interpolation_values = np.linspace(
            0.0,
            1.0,
            NUM_FRAMES,
        )

        volumes = [
            torch.sigmoid(
                model.decode((1.0 - t) * za + t * zb)
            ).cpu().numpy()[0, 0]
            for t in interpolation_values
        ]

    output_dir = (
        ROOT / "outputs" / "experiments" / "interpolation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, volume in enumerate(volumes):
        np.save(output_dir / f"{index:03d}.npy", volume)

    render_animation(
        volumes,
        ROOT / "docs" / "images" / "latent_interpolation.gif",
    )


if __name__ == "__main__":
    main()