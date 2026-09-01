from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import TrophyDataset
from src.models.autoencoder import Autoencoder3D
from src.models.vae import VAE3D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 32
BATCH_SIZE = 32
THRESHOLD = 0.5
NUM_VISUAL_EXAMPLES = 8

AE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "ae_best.pt"
VAE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "vae_best.pt"
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "ae_vs_vae"
RENDER_DIR = ROOT / "outputs" / "renders"

def binary_iou(prediction, target):
    prediction = (prediction >= THRESHOLD)

    target = (target >= THRESHOLD)

    intersection = (
        prediction & target
    ).sum(
        dim=(1, 2, 3, 4)
    ).float()

    union = (
        prediction | target
    ).sum(
        dim=(1, 2, 3, 4)
    ).float()

    iou = (
        intersection
        / union.clamp_min(1.0)
    )

    return iou

def dice_score(prediction, target):

    prediction = (prediction >= THRESHOLD)

    target = (
        target >= THRESHOLD
    )

    intersection = (
        prediction & target
    ).sum(
        dim=(1, 2, 3, 4)
    ).float()

    prediction_count = (
        prediction.sum(
            dim=(1, 2, 3, 4)
        ).float()
    )

    target_count = (
        target.sum(
            dim=(1, 2, 3, 4)
        ).float()
    )

    dice = (
        2.0 * intersection
        / (
            prediction_count + target_count
        ).clamp_min(1.0)
    )

    return dice

def load_models():

    ae = Autoencoder3D(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    ae.load_state_dict(
        torch.load(
            AE_CHECKPOINT,
            map_location=DEVICE,
        )
    )

    ae.eval()

    vae = VAE3D(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    vae.load_state_dict(
        torch.load(
            VAE_CHECKPOINT,
            map_location=DEVICE,
        )
    )

    vae.eval()

    return ae, vae

def evaluate_reconstruction(ae, vae, test_loader):
    ae_bce = []
    vae_bce = []

    ae_iou = []
    vae_iou = []

    ae_dice = []
    vae_dice = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                voxels = batch[0]
            else:
                voxels = batch

            voxels = voxels.to(
                DEVICE
            )

            ae_logits = ae(
                voxels
            )

            ae_probabilities = (
                torch.sigmoid(
                    ae_logits
                )
            )

            ae_batch_bce = (
                F.binary_cross_entropy_with_logits(
                    ae_logits,
                    voxels,
                    reduction="none",
                )
                .flatten(1)
                .mean(dim=1)
            )

            ae_bce.extend(
                ae_batch_bce
                .cpu()
                .numpy()
                .tolist()
            )

            ae_iou.extend(
                binary_iou(
                    ae_probabilities,
                    voxels,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            ae_dice.extend(
                dice_score(
                    ae_probabilities,
                    voxels,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            mu, logvar = vae.encode(
                voxels
            )

            vae_logits = vae.decode(
                mu
            )

            vae_probabilities = (
                torch.sigmoid(
                    vae_logits
                )
            )

            vae_batch_bce = (
                F.binary_cross_entropy_with_logits(
                    vae_logits,
                    voxels,
                    reduction="none",
                )
                .flatten(1)
                .mean(dim=1)
            )

            vae_bce.extend(
                vae_batch_bce
                .cpu()
                .numpy()
                .tolist()
            )

            vae_iou.extend(
                binary_iou(
                    vae_probabilities,
                    voxels,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            vae_dice.extend(
                dice_score(
                    vae_probabilities,
                    voxels,
                )
                .cpu()
                .numpy()
                .tolist()
            )

    return {
        "ae": {
            "bce": float(
                np.mean(ae_bce)
            ),
            "iou": float(
                np.mean(ae_iou)
            ),
            "dice": float(
                np.mean(ae_dice)
            ),
        },

        "vae": {
            "bce": float(
                np.mean(vae_bce)
            ),
            "iou": float(
                np.mean(vae_iou)
            ),
            "dice": float(
                np.mean(vae_dice)
            ),
        },
    }

def visualize_reconstructions(ae, vae, test_dataset):
    num_examples = min(
        NUM_VISUAL_EXAMPLES,
        len(test_dataset),
    )

    fig = plt.figure(
        figsize=(
            3 * num_examples,
            9,
        )
    )

    for i in range(num_examples):
        sample = test_dataset[i]

        if isinstance(sample, (list, tuple)):
            voxels = sample[0]
        else:
            voxels = sample

        x = (
            voxels
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            ae_logits = ae(x)

            ae_probabilities = (
                torch.sigmoid(
                    ae_logits
                )
            )

            mu, logvar = vae.encode(
                x
            )

            vae_logits = vae.decode(
                mu
            )

            vae_probabilities = (
                torch.sigmoid(
                    vae_logits
                )
            )

        original = (
            x[0, 0]
            .cpu()
            .numpy()
            >= THRESHOLD
        )

        ae_reconstruction = (
            ae_probabilities[
                0,
                0,
            ]
            .cpu()
            .numpy()
            >= THRESHOLD
        )

        vae_reconstruction = (
            vae_probabilities[
                0,
                0,
            ]
            .cpu()
            .numpy()
            >= THRESHOLD
        )

        volumes = [
            original,
            ae_reconstruction,
            vae_reconstruction,
        ]

        row_names = [
            "Original",
            "AE",
            "VAE",
        ]

        for row in range(3):
            plot_index = (
                row
                * num_examples
                + i
                + 1
            )

            ax = fig.add_subplot(
                3,
                num_examples,
                plot_index,
                projection="3d",
            )

            ax.voxels(
                volumes[row]
            )

            if i == 0:
                ax.text2D(
                    -0.25,
                    0.5,
                    row_names[row],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=11,
                )

            if row == 0:
                ax.set_title(
                    f"Test {i}",
                    fontsize=10,
                )

            ax.set_axis_off()

            ax.set_box_aspect(
                (1, 1, 1)
            )

    fig.suptitle(
        "AE vs VAE Reconstruction",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        RENDER_DIR
        / "ae_vs_vae_reconstruction.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )

def load_vae_generation_results():

    experiments = {
        "std_1.0": (
            ROOT
            / "outputs"
            / "experiments"
            / "vae_samples"
            / "summary.json"
        ),

        "std_0.7": (
            ROOT
            / "outputs"
            / "experiments"
            / "vae_samples_std_0.7"
            / "summary.json"
        ),

        "std_0.5": (
            ROOT
            / "outputs"
            / "experiments"
            / "vae_samples_std_0.5"
            / "summary.json"
        ),
    }

    results = {}

    for name, path in (experiments.items()):
        if not path.exists():
            print(
                f"Warning: missing "
                f"{path}"
            )
            continue

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            results[name] = (
                json.load(file)
            )

    return results

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RENDER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Using device: {DEVICE}"
    )

    dataset = TrophyDataset()
    
    generator = torch.Generator().manual_seed(
        42
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = int(
        0.1 * len(dataset)
    )

    test_size = (
        len(dataset)
        - train_size
        - val_size
    )

    train_dataset, val_dataset, test_dataset = (
        torch.utils.data.random_split(
            dataset,
            [
                train_size,
                val_size,
                test_size,
            ],
            generator=generator,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    ae, vae = load_models()

    reconstruction = (
        evaluate_reconstruction(
            ae,
            vae,
            test_loader,
        )
    )

    print()
    print(
        "================================"
    )
    print(
        "AE vs VAE RECONSTRUCTION"
    )
    print(
        "================================"
    )

    print()
    print("AE")
    print(
        f"  BCE:  "
        f"{reconstruction['ae']['bce']:.6f}"
    )
    print(
        f"  IoU:  "
        f"{reconstruction['ae']['iou']:.4f}"
    )
    print(
        f"  Dice: "
        f"{reconstruction['ae']['dice']:.4f}"
    )

    print()
    print("VAE")
    print(
        f"  BCE:  "
        f"{reconstruction['vae']['bce']:.6f}"
    )
    print(
        f"  IoU:  "
        f"{reconstruction['vae']['iou']:.4f}"
    )
    print(
        f"  Dice: "
        f"{reconstruction['vae']['dice']:.4f}"
    )

    vae_generation = (
        load_vae_generation_results()
    )

    combined_results = {
        "reconstruction": reconstruction,
        "vae_generation": vae_generation,
    }

    json_path = (
        OUTPUT_DIR
        / "comparison.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_results,
            file,
            indent=2,
        )

    print()
    print(
        f"Saved: {json_path}"
    )

    csv_path = (
        OUTPUT_DIR
        / "reconstruction_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "model",
                "bce",
                "iou",
                "dice",
            ]
        )

        for model_name in [
            "ae",
            "vae",
        ]:
            writer.writerow(
                [
                    model_name,
                    reconstruction[
                        model_name
                    ]["bce"],
                    reconstruction[
                        model_name
                    ]["iou"],
                    reconstruction[
                        model_name
                    ]["dice"],
                ]
            )

    print(
        f"Saved: {csv_path}"
    )

    visualize_reconstructions(
        ae,
        vae,
        test_dataset,
    )


if __name__ == "__main__":
    main()