from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import get_loaders
from src.models.vae import VAE3D


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    ROOT / "outputs" / "checkpoints" / "vae_best.pt"
)


def main():
    train_loader, _, _, _, _, _ = get_loaders(
        batch_size=32
    )

    model = VAE3D(latent_dim=32).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )
    )

    model.eval()

    mus = []
    logvars = []

    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(DEVICE)

            mu, logvar = model.encode(x)

            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

    mus = np.concatenate(mus, axis=0)
    logvars = np.concatenate(logvars, axis=0)

    stds = np.exp(0.5 * logvars)

    print("Encoded latent statistics")
    print("-------------------------")
    print(f"mu mean:       {mus.mean():.4f}")
    print(f"mu std:        {mus.std():.4f}")
    print(f"mu min:        {mus.min():.4f}")
    print(f"mu max:        {mus.max():.4f}")
    print()
    print(f"posterior std mean: {stds.mean():.4f}")
    print(f"posterior std min:  {stds.min():.4f}")
    print(f"posterior std max:  {stds.max():.4f}")

    print("\nPer-dimension:")
    print("dim    mu_mean    mu_std    posterior_std")

    for i in range(mus.shape[1]):
        print(
            f"{i:02d}   "
            f"{mus[:, i].mean():8.3f}   "
            f"{mus[:, i].std():8.3f}   "
            f"{stds[:, i].mean():8.3f}"
        )

if __name__ == "__main__":
    main()