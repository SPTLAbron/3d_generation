from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.vae import VAE3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = ROOT / "outputs" / "checkpoints" / "vae_best.pt"
OUTPUT_DIR =  ROOT / "outputs" / "experiments" / "vae_samples_std_0.5"

LATENT_DIM = 32
NUM_SAMPLES = 100
SAMPLE_STD = 0.5
SEED = 42

def main():

    print(f"Using device: {DEVICE}")

    print(f"Sampling std: {SAMPLE_STD}")

    print(f"Number of samples: {NUM_SAMPLES}")

    model = VAE3D(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
        )
    )

    model.eval()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in OUTPUT_DIR.glob("sample_*.npy"):
        old_file.unlink()

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    z = (
        SAMPLE_STD
        * torch.randn(
            NUM_SAMPLES,
            LATENT_DIM,
            device=DEVICE,
        )
    )

    print(
        f"Latent tensor shape: "
        f"{tuple(z.shape)}"
    )

    print(
        f"Actual latent mean: "
        f"{z.mean().item():.4f}"
    )

    print(
        f"Actual latent std: "
        f"{z.std().item():.4f}"
    )

    with torch.no_grad():
        logits = model.decode(z)

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

    print(
        f"Decoded tensor shape: "
        f"{tuple(probabilities.shape)}"
    )

    expected_shape = (
        NUM_SAMPLES,
        1,
        32,
        32,
        32,
    )

    if tuple(probabilities.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected decoder output shape. "
            f"Expected {expected_shape}, "
            f"got "
            f"{tuple(probabilities.shape)}"
        )

    for i in range(NUM_SAMPLES):
        voxels = (
            probabilities[
                i,
                0,
            ]
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        np.save(
            OUTPUT_DIR
            / f"sample_{i:03d}.npy",
            voxels,
        )

    saved_files = list(
        OUTPUT_DIR.glob(
            "sample_*.npy"
        )
    )

    if len(saved_files) != NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} "
            f"saved samples, but found "
            f"{len(saved_files)}."
        )

    print()
    print(
        f"Generated "
        f"{len(saved_files)} "
        f"VAE samples."
    )

    print(
        f"Saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()