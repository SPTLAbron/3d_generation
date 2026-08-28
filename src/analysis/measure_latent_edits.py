from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    get_loaders,
    PARAM_COLUMNS,
)
from src.models.autoencoder import (
    Autoencoder3D,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "ae_best.pt"
)

LATENT_EDIT_ROOT = (
    ROOT
    / "outputs"
    / "experiments"
    / "latent_edits"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "experiments"
    / "disentanglement.csv"
)

MATRIX_PATH = (
    ROOT
    / "outputs"
    / "experiments"
    / "disentanglement_matrix.csv"
)


def encode_loader(
    model,
    loader,
):
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
        np.concatenate(
            zs,
            axis=0,
        ),
        np.concatenate(
            ys,
            axis=0,
        ),
    )


def train_parameter_probes(
    z_train,
    y_train,
):
    probes = {}

    for i, parameter_name in enumerate(
        PARAM_COLUMNS
    ):
        reg = LinearRegression()

        reg.fit(
            z_train,
            y_train[:, i],
        )

        probes[
            parameter_name
        ] = reg

    return probes


def predict_parameters(
    probes,
    z,
):
    z = np.asarray(
        z,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    return {
        parameter_name:
            float(
                reg.predict(z)[0]
            )
        for (
            parameter_name,
            reg
        ) in probes.items()
    }


def main():
    (
        train_loader,
        _,
        _,
        _,
        _,
        _,
    ) = get_loaders(
        batch_size=32
    )

    model = Autoencoder3D(
        latent_dim=32
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
        )
    )

    model.eval()

    print(
        "Encoding training set..."
    )

    z_train, y_train = encode_loader(
        model,
        train_loader,
    )

    print(
        "Training parameter probes..."
    )

    probes = train_parameter_probes(
        z_train,
        y_train,
    )

    parameter_stds = {
        name: float(
            np.std(
                y_train[:, i]
            )
        )
        for i, name in enumerate(
            PARAM_COLUMNS
        )
    }

    rows = []

    for edit_parameter in PARAM_COLUMNS:
        edit_dir = (
            LATENT_EDIT_ROOT
            / edit_parameter
        )

        if not edit_dir.exists():
            print(
                "Skipping missing edit: "
                f"{edit_parameter}"
            )
            continue

        baseline_path = (
            edit_dir
            / "alpha_+0.0_latent.npy"
        )

        if not baseline_path.exists():
            print(
                "Missing baseline: "
                f"{baseline_path}"
            )
            continue

        baseline_z = np.load(
            baseline_path
        )

        baseline_predictions = (
            predict_parameters(
                probes,
                baseline_z,
            )
        )

        for latent_path in sorted(
            edit_dir.glob(
                "alpha_*_latent.npy"
            )
        ):
            name = latent_path.stem

            alpha_string = (
                name
                .replace(
                    "alpha_",
                    ""
                )
                .replace(
                    "_latent",
                    ""
                )
            )

            alpha = float(
                alpha_string
            )

            z_edit = np.load(
                latent_path
            )

            predictions = (
                predict_parameters(
                    probes,
                    z_edit,
                )
            )

            for measured_parameter in (
                PARAM_COLUMNS
            ):
                baseline_value = (
                    baseline_predictions[
                        measured_parameter
                    ]
                )

                edited_value = (
                    predictions[
                        measured_parameter
                    ]
                )

                delta = (
                    edited_value
                    - baseline_value
                )

                std = (
                    parameter_stds[
                        measured_parameter
                    ]
                )

                if std > 1e-8:
                    normalized_delta = (
                        delta / std
                    )
                else:
                    normalized_delta = (
                        np.nan
                    )

                rows.append(
                    {
                        "edit_parameter":
                            edit_parameter,
                        "alpha":
                            alpha,
                        "measured_parameter":
                            measured_parameter,
                        "baseline_prediction":
                            baseline_value,
                        "edited_prediction":
                            edited_value,
                        "delta":
                            delta,
                        "normalized_delta":
                            normalized_delta,
                    }
                )

    df = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved full results: "
        f"{OUTPUT_PATH}"
    )
    
    positive_edit_df = df[
        np.isclose(
            df["alpha"],
            2.0,
        )
    ]

    matrix = (
        positive_edit_df
        .pivot(
            index="edit_parameter",
            columns="measured_parameter",
            values="normalized_delta",
        )
        .reindex(
            index=PARAM_COLUMNS,
            columns=PARAM_COLUMNS,
        )
    )

    matrix.to_csv(
        MATRIX_PATH
    )

    print(
        f"Saved summary matrix: "
        f"{MATRIX_PATH}"
    )

    print()
    print(
        "Normalized parameter changes "
        "for alpha = +2:"
    )

    print()

    print(
        matrix.round(3)
    )
    
    print()
    print("DISENTANGLEMENT RATIOS")
    print("-" * 70)

    ratio_rows = []

    for edit_parameter in PARAM_COLUMNS:
        if edit_parameter not in matrix.index:
            continue

        row = matrix.loc[
            edit_parameter
        ]

        diagonal = abs(
            row[
                edit_parameter
            ]
        )

        off_diagonal = (
            row.drop(
                labels=[edit_parameter]
            )
            .abs()
            .dropna()
        )

        if len(off_diagonal) == 0:
            continue

        mean_cross_effect = (
            off_diagonal.mean()
        )

        if mean_cross_effect < 1e-8:
            ratio = np.inf
        else:
            ratio = (
                diagonal
                / mean_cross_effect
            )

        ratio_rows.append(
            {
                "edit_parameter":
                    edit_parameter,
                "target_effect":
                    diagonal,
                "mean_cross_effect":
                    mean_cross_effect,
                "disentanglement_ratio":
                    ratio,
            }
        )

        print(
            f"{edit_parameter:25s} "
            f"target={diagonal:8.3f} "
            f"cross={mean_cross_effect:8.3f} "
            f"ratio={ratio:8.3f}"
        )


    ratio_path = (
        ROOT
        / "outputs"
        / "experiments"
        / "disentanglement_ratios.csv"
    )

    pd.DataFrame(
        ratio_rows
    ).to_csv(
        ratio_path,
        index=False,
    )

    print()
    print(
        f"Saved ratios: {ratio_path}"
    )


if __name__ == "__main__":
    main()