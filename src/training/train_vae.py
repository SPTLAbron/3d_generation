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

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

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

            batch_size = len(x)

            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kl += kl.item() * batch_size

    dataset_size = len(loader.dataset)

    return (
        total_loss / dataset_size,
        total_recon / dataset_size,
        total_kl / dataset_size,
    )


def train(
    epochs=30,
    batch_size=16,
    lr=1e-3,
    beta=1e-3,
    warmup_epochs=20,
    patience=5,
    min_improvement=1e-4,
):
    print(f"Using device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Maximum beta: {beta}")
    print(f"KL warmup epochs: {warmup_epochs}")

    train_loader, val_loader, _, _, _, _ = get_loaders(
        batch_size
    )

    model = VAE3D(
        latent_dim=32
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_warmup_recon = float("inf")

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        current_beta = beta * min(
            1.0,
            epoch / warmup_epochs,
        )

        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            current_batch_size = len(x)

            total_loss += (
                loss.item() * current_batch_size
            )
            total_recon += (
                recon.item() * current_batch_size
            )
            total_kl += (
                kl.item() * current_batch_size
            )

        dataset_size = len(train_loader.dataset)

        train_loss = total_loss / dataset_size
        train_recon = total_recon / dataset_size
        train_kl = total_kl / dataset_size

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

        if epoch < warmup_epochs:
            if val_recon < best_warmup_recon:
                best_warmup_recon = val_recon

                torch.save(
                    model.state_dict(),
                    CHECKPOINT_DIR / "vae_warmup_best.pt",
                )

                print(
                    "  saved best warmup reconstruction"
                )

        else:
            if val_loss < best_val_loss - min_improvement:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    CHECKPOINT_DIR / "vae_best.pt",
                )

                print("  saved new best VAE")
            else:
                epochs_without_improvement += 1

                print(
                    "  epochs without improvement: "
                    f"{epochs_without_improvement}/"
                    f"{patience}"
                )

            if val_kl < 0.05:
                print(
                    "Warning: validation KL is close to "
                    "zero, indicating possible posterior "
                    "collapse."
                )

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"validation loss did not improve by "
                    f"at least {min_improvement} for "
                    f"{patience} epochs."
                )
                break

    final_checkpoint = CHECKPOINT_DIR / "vae_best.pt"

    if final_checkpoint.exists():
        model.load_state_dict(
            torch.load(
                final_checkpoint,
                map_location=DEVICE,
            )
        )

        print(
            f"Loaded best checkpoint: "
            f"{final_checkpoint}"
        )
    else:
        fallback_checkpoint = (
            CHECKPOINT_DIR / "vae_warmup_best.pt"
        )

        if fallback_checkpoint.exists():
            model.load_state_dict(
                torch.load(
                    fallback_checkpoint,
                    map_location=DEVICE,
                )
            )

            print(
                "No post-warmup checkpoint was created. "
                f"Loaded fallback: {fallback_checkpoint}"
            )

    return model


if __name__ == "__main__":
    train()