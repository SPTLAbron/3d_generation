from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import get_loaders
from src.models.autoencoder import Autoencoder3D


DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = ROOT / "outputs" / "checkpoints" / "ae_best.pt"
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "reconstructions"


def main():
    _, _, test_loader, _, _, _ = get_loaders(batch_size=1)

    model = Autoencoder3D(latent_dim=32).to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE)
    )
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_loss = 0.0

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits, x)

            total_loss += loss.item()

            reconstruction = torch.sigmoid(logits)

            original = x[0, 0].cpu().numpy()
            reconstructed = reconstruction[0, 0].cpu().numpy()

            np.save(
                OUTPUT_DIR / f"{i:03d}_original.npy",
                original,
            )

            np.save(
                OUTPUT_DIR / f"{i:03d}_reconstruction.npy",
                reconstructed,
            )

    mean_loss = total_loss / len(test_loader)

    print(f"Test BCE loss: {mean_loss:.6f}")
    print(f"Saved reconstructions to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()