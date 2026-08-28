from pathlib import Path
import sys

import numpy as np
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

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "experiments"
    / "shape_optimization"
)

LATENT_DIM = 32

NUM_STEPS = 30
LEARNING_RATE = 0.05

LATENT_REGULARIZATION = 0.02

LATENT_BOUND_STD = 3.0

BASE_PARAMETER = "lower_base_radius"
BALL_PARAMETER = "ball_radius"

def encode_loader(
    model,
    loader,
):
    """
    Encode all samples from a DataLoader.

    Returns
    -------
    z : np.ndarray
        Latent vectors with shape [N, latent_dim].

    y : np.ndarray
        Procedural parameters with shape
        [N, number_of_parameters].
    """

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


def fit_probe(
    z_train,
    y_train,
    parameter_name,
):
    """
    Fit a linear regression probe that predicts one
    procedural parameter from the latent representation.

    Returns the regression weights and bias as tensors
    so that the prediction remains differentiable with
    respect to z.
    """

    index = PARAM_COLUMNS.index(
        parameter_name
    )

    reg = LinearRegression()

    reg.fit(
        z_train,
        y_train[:, index],
    )

    weight = torch.tensor(
        reg.coef_,
        dtype=torch.float32,
        device=DEVICE,
    )

    bias = torch.tensor(
        reg.intercept_,
        dtype=torch.float32,
        device=DEVICE,
    )

    return weight, bias


def probe_prediction(
    z,
    weight,
    bias,
):
    """
    Differentiable linear-probe prediction.
    """

    return (
        z @ weight
        + bias
    )


def main():

    print(
        f"Using device: {DEVICE}"
    )

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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        "Encoding training set..."
    )

    z_train, y_train = encode_loader(
        model,
        train_loader,
    )

    print(
        f"Encoded training samples: "
        f"{len(z_train)}"
    )

    latent_mean = torch.tensor(
        z_train.mean(axis=0),
        dtype=torch.float32,
        device=DEVICE,
    )

    latent_std = torch.tensor(
        z_train.std(axis=0),
        dtype=torch.float32,
        device=DEVICE,
    ).clamp_min(
        1e-6
    )

    latent_min = (
        latent_mean
        - LATENT_BOUND_STD
        * latent_std
    )

    latent_max = (
        latent_mean
        + LATENT_BOUND_STD
        * latent_std
    )

    print(
        f"Fitting probe for "
        f"{BASE_PARAMETER}..."
    )

    (
        base_weight,
        base_bias,
    ) = fit_probe(
        z_train,
        y_train,
        BASE_PARAMETER,
    )

    print(
        f"Fitting probe for "
        f"{BALL_PARAMETER}..."
    )

    (
        ball_weight,
        ball_bias,
    ) = fit_probe(
        z_train,
        y_train,
        BALL_PARAMETER,
    )

    base_index = PARAM_COLUMNS.index(
        BASE_PARAMETER
    )

    ball_index = PARAM_COLUMNS.index(
        BALL_PARAMETER
    )

    base_mean = torch.tensor(
        y_train[
            :,
            base_index
        ].mean(),
        dtype=torch.float32,
        device=DEVICE,
    )

    base_std = torch.tensor(
        y_train[
            :,
            base_index
        ].std(),
        dtype=torch.float32,
        device=DEVICE,
    ).clamp_min(
        1e-6
    )

    ball_mean = torch.tensor(
        y_train[
            :,
            ball_index
        ].mean(),
        dtype=torch.float32,
        device=DEVICE,
    )

    ball_std = torch.tensor(
        y_train[
            :,
            ball_index
        ].std(),
        dtype=torch.float32,
        device=DEVICE,
    ).clamp_min(
        1e-6
    )

    print()
    print(
        "Training parameter statistics:"
    )

    print(
        f"{BASE_PARAMETER}: "
        f"mean={base_mean.item():.4f}, "
        f"std={base_std.item():.4f}"
    )

    print(
        f"{BALL_PARAMETER}: "
        f"mean={ball_mean.item():.4f}, "
        f"std={ball_std.item():.4f}"
    )

    x, original_parameters = (
        test_dataset[0]
    )

    original_base = (
        original_parameters[
            base_index
        ].item()
    )

    original_ball = (
        original_parameters[
            ball_index
        ].item()
    )

    print()
    print(
        "Starting test trophy:"
    )

    print(
        f"Ground-truth "
        f"{BASE_PARAMETER}: "
        f"{original_base:.4f}"
    )

    print(
        f"Ground-truth "
        f"{BALL_PARAMETER}: "
        f"{original_ball:.4f}"
    )

    x = (
        x
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        z0 = (
            model.encode(x)
            .detach()
        )

    z = (
        z0.clone()
        .detach()
        .requires_grad_(True)
    )

    optimizer = torch.optim.Adam(
        [z],
        lr=LEARNING_RATE,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in OUTPUT_DIR.glob(
        "step_*.npy"
    ):
        old_file.unlink()

    history_path = (
        OUTPUT_DIR
        / "history.csv"
    )

    if history_path.exists():
        history_path.unlink()

    print()
    print("Starting parameter comparison:")
    print(
        f"  lower_base_radius: "
        f"ground_truth={original_base:.4f}, "
        f"probe={probe_prediction(z0, base_weight, base_bias).item():.4f}"
    )
    print(
        f"  ball_radius: "
        f"ground_truth={original_ball:.4f}, "
        f"probe={probe_prediction(z0, ball_weight, ball_bias).item():.4f}"
    )

    history = []
    bound_hits = 0
    
    print()
    print(
        "Starting latent optimization..."
    )

    print(
        "-" * 90
    )

    for step in range(
        NUM_STEPS + 1
    ):

        predicted_base = (
            probe_prediction(
                z,
                base_weight,
                base_bias,
            )
        )

        predicted_ball = (
            probe_prediction(
                z,
                ball_weight,
                ball_bias,
            )
        )

        normalized_base = (
            predicted_base
            - base_mean
        ) / base_std

        normalized_ball = (
            predicted_ball
            - ball_mean
        ) / ball_std

        latent_distance = (
            (z - z0)
            .pow(2)
            .mean()
        )

        objective = (
            -normalized_base.mean()
            + normalized_ball.mean()
            + LATENT_REGULARIZATION
            * latent_distance
        )

        with torch.no_grad():

            logits = model.decode(
                z
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            voxels = (
                probabilities[
                    0,
                    0,
                ]
                .cpu()
                .numpy()
            )

            latent_np = (
                z[
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )

            np.save(
                OUTPUT_DIR
                / f"step_{step:03d}.npy",
                voxels,
            )

            np.save(
                OUTPUT_DIR
                / (
                    f"step_{step:03d}"
                    "_latent.npy"
                ),
                latent_np,
            )

            history.append(
                [
                    step,
                    float(
                        predicted_base.item()
                    ),
                    float(
                        predicted_ball.item()
                    ),
                    float(
                        normalized_base.item()
                    ),
                    float(
                        normalized_ball.item()
                    ),
                    float(
                        latent_distance.item()
                    ),
                    float(
                        objective.item()
                    ),
                    int(
                        bound_hits
                    ),
                ]
            )

        print(
            f"step={step:03d} "
            f"base={predicted_base.item():.4f} "
            f"ball={predicted_ball.item():.4f} "
            f"base_z={normalized_base.item():+.3f} "
            f"ball_z={normalized_ball.item():+.3f} "
            f"distance={latent_distance.item():.4f} "
            f"objective={objective.item():.4f}"
        )

        if step == NUM_STEPS:
            break
        
        optimizer.zero_grad()

        objective.backward()

        optimizer.step()

        with torch.no_grad():
            z.clamp_(
                latent_min.unsqueeze(0),
                latent_max.unsqueeze(0),
            )

            lower_hits = torch.isclose(
                z,
                latent_min.unsqueeze(0),
                atol=1e-5,
            ).sum().item()

            upper_hits = torch.isclose(
                z,
                latent_max.unsqueeze(0),
                atol=1e-5,
            ).sum().item()

            bound_hits = (
                lower_hits
                + upper_hits
            )

            if bound_hits > 0:
                print(
                    f"  latent dimensions at bounds: "
                    f"{bound_hits}/{LATENT_DIM}"
                )

    history = np.asarray(
        history,
        dtype=np.float32,
    )

    np.savetxt(
        history_path,
        history,
        delimiter=",",
        header=(
            "step,"
            "predicted_lower_base_radius,"
            "predicted_ball_radius,"
            "normalized_lower_base_radius,"
            "normalized_ball_radius,"
            "latent_distance,"
            "objective,"
            "latent_bound_hits"
        ),
        comments="",
    )

    initial_base = history[
        0,
        1
    ]

    final_base = history[
        -1,
        1
    ]

    initial_ball = history[
        0,
        2
    ]

    final_ball = history[
        -1,
        2
    ]

    print(
        "-" * 90
    )

    print()
    print(
        "Optimization complete."
    )

    print()

    print(
        f"{BASE_PARAMETER}: "
        f"{initial_base:.4f} "
        f"-> {final_base:.4f} "
        f"(change "
        f"{final_base - initial_base:+.4f})"
    )

    print(
        f"{BALL_PARAMETER}: "
        f"{initial_ball:.4f} "
        f"-> {final_ball:.4f} "
        f"(change "
        f"{final_ball - initial_ball:+.4f})"
    )

    print(
        f"Final latent distance: "
        f"{history[-1, 5]:.4f}"
    )

    print()

    print(
        f"Saved optimization to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        f"History:"
    )

    print(
        history_path
    )


if __name__ == "__main__":
    main()