from pathlib import Path
import csv
import sys

import numpy as np
import torch
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import get_loaders, PARAM_COLUMNS
from src.models.autoencoder import Autoencoder3D


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "ae_best.pt"
)

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "experiments"
    / "latent_edits"
)

LATENT_DIM = 32

EDIT_PARAMETERS = [
    "ball_radius",
    "support_sweep",
    "body_height",
    "body_bottom_radius",
    "body_top_radius",
    "lower_base_radius",
]

ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def encode_loader(model, loader):
    zs = []
    ys = []

    with torch.no_grad():
        for x, params in loader:
            x = x.to(DEVICE)

            z = model.encode(x)

            zs.append(
                z.cpu().numpy()
            )

            ys.append(
                params.numpy()
            )

    return (
        np.concatenate(zs, axis=0),
        np.concatenate(ys, axis=0),
    )


def fit_probe(
    z_train,
    y_train,
    parameter_index,
):
    reg = LinearRegression()

    reg.fit(
        z_train,
        y_train[:, parameter_index],
    )

    direction = reg.coef_.astype(
        np.float32
    )

    norm = np.linalg.norm(direction)

    if norm < 1e-8:
        raise ValueError(
            "Probe direction has near-zero norm."
        )

    direction /= norm

    return reg, direction


def main():
    (
        train_loader,
        _,
        _,
        _,
        _,
        test_dataset,
    ) = get_loaders(
        batch_size=32
    )

    model = Autoencoder3D(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
        )
    )

    model.eval()

    print(
        f"Using device: {DEVICE}"
    )

    print(
        "Encoding training set..."
    )

    z_train, y_train = encode_loader(
        model,
        train_loader,
    )

    x, original_parameters = (
        test_dataset[0]
    )

    x = (
        x
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        z_original = model.encode(x)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for parameter_name in EDIT_PARAMETERS:
        parameter_index = (
            PARAM_COLUMNS.index(
                parameter_name
            )
        )

        reg, direction_np = fit_probe(
            z_train,
            y_train,
            parameter_index,
        )

        direction = torch.tensor(
            direction_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        output_dir = (
            OUTPUT_ROOT
            / parameter_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_dir
            / "direction.npy",
            direction_np,
        )

        original_value = (
            original_parameters[
                parameter_index
            ].item()
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"Editing: {parameter_name}"
        )

        print(
            f"Original procedural value: "
            f"{original_value:.6f}"
        )

        print(
            "=" * 70
        )

        with torch.no_grad():
            for alpha in ALPHAS:
                z_edit = (
                    z_original
                    + alpha
                    * direction.unsqueeze(0)
                )

                logits = model.decode(
                    z_edit
                )

                probabilities = (
                    torch.sigmoid(logits)
                )

                voxels = (
                    probabilities[
                        0,
                        0,
                    ]
                    .cpu()
                    .numpy()
                )

                z_edit_np = (
                    z_edit[
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                alpha_name = (
                    f"{alpha:+.1f}"
                )

                voxel_path = (
                    output_dir
                    / (
                        f"alpha_"
                        f"{alpha_name}_voxels.npy"
                    )
                )

                latent_path = (
                    output_dir
                    / (
                        f"alpha_"
                        f"{alpha_name}_latent.npy"
                    )
                )

                np.save(
                    voxel_path,
                    voxels,
                )

                np.save(
                    latent_path,
                    z_edit_np,
                )

                predicted_value = (
                    reg.predict(
                        z_edit_np[
                            None,
                            :
                        ]
                    )[0]
                )

                manifest_rows.append(
                    {
                        "edit_parameter":
                            parameter_name,
                        "alpha":
                            alpha,
                        "original_value":
                            original_value,
                        "probe_predicted_value":
                            predicted_value,
                        "voxel_file":
                            str(
                                voxel_path.relative_to(
                                    ROOT
                                )
                            ),
                        "latent_file":
                            str(
                                latent_path.relative_to(
                                    ROOT
                                )
                            ),
                    }
                )

                print(
                    f"alpha={alpha:+.1f} "
                    f"probe_prediction="
                    f"{predicted_value:.6f}"
                )

    manifest_path = (
        OUTPUT_ROOT
        / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "edit_parameter",
            "alpha",
            "original_value",
            "probe_predicted_value",
            "voxel_file",
            "latent_file",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    print()
    print(
        "Latent edits complete."
    )

    print(
        f"Saved to: {OUTPUT_ROOT}"
    )

    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()