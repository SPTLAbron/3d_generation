from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import get_loaders
from src.models.vae import VAE3D


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"


def vae_loss(logits, x, mu, logvar, beta=1e-3):
    recon = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        x,
        reduction="mean",
    )

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total = recon + beta * kl

    return total, recon, kl


def evaluate(model, loader, beta):
    model.eval()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)

            logits, mu, logvar = model(x)

            loss, recon, kl = vae_loss(
                logits,
                x,
                mu,
                logvar,
                beta,
            )

            n = len(x)

            total_loss += loss.item() * n
            total_recon += recon.item() * n
            total_kl += kl.item() * n

    n = len(loader.dataset)

    return (
        total_loss / n,
        total_recon / n,
        total_kl / n,
    )


def train(
    epochs=100,
    batch_size=16,
    lr=1e-3,
    beta=1e-2,
):
    train_loader, val_loader, _, _, _, _ = get_loaders(
        batch_size
    )

    model = VAE3D(latent_dim=32).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        
        warmup_epochs = 20

        current_beta = beta * min(
            1.0,
            epoch / warmup_epochs
        )

        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0

        for x, _ in train_loader:

            x = x.to(DEVICE)

            optimizer.zero_grad()

            logits, mu, logvar = model(x)

            loss, recon, kl = vae_loss(
                logits,
                x,
                mu,
                logvar,
                current_beta,
            )

            loss.backward()
            optimizer.step()

            n = len(x)

            total_loss += loss.item() * n
            total_recon += recon.item() * n
            total_kl += kl.item() * n

        n = len(train_loader.dataset)

        train_loss = total_loss / n
        train_recon = total_recon / n
        train_kl = total_kl / n

        val_loss, val_recon, val_kl = evaluate(
            model,
            val_loader,
            current_beta,
        )

        print(
            f"{epoch:03d} "
            f"beta={current_beta:.6f} "
            f"train={train_loss:.6f} "
            f"recon={train_recon:.6f} "
            f"kl={train_kl:.6f} "
            f"val={val_loss:.6f} "
            f"val_recon={val_recon:.6f} "
            f"val_kl={val_kl:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "vae_best.pt",
            )

            print("  saved new best VAE")

    return model


if __name__ == "__main__":
    train()