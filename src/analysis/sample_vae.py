from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.vae import VAE3D


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "vae_best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "experiments"
    / "vae_samples"
)


def main(num_samples=100, latent_dim=32):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = VAE3D(
        latent_dim=latent_dim
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )
    )

    model.eval()

    with torch.no_grad():
        torch.manual_seed(42)
        
        z = torch.randn(
            num_samples,
            latent_dim,
            device=DEVICE
        )

        logits = model.decode(z)

        probabilities = torch.sigmoid(logits)

        for i in range(num_samples):

            voxel_grid = (
                probabilities[i, 0]
                .cpu()
                .numpy()
            )

            np.save(
                OUTPUT_DIR / f"sample_{i:03d}.npy",
                voxel_grid
            )

    print(
        f"Generated {num_samples} VAE samples."
    )

    print(
        f"Saved to: {OUTPUT_DIR}"
    )

if __name__ == "__main__":
    main()