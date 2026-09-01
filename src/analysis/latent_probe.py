from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.data.dataset import get_loaders,PARAM_COLUMNS
from src.models.autoencoder import Autoencoder3D

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_loader(model, loader):
    zs = []
    ys = []

    with torch.no_grad():
        for x, params in loader:
            x = x.to(DEVICE)

            z = model.encode(x)

            zs.append(z.cpu().numpy())
            ys.append(params.numpy())

    return np.concatenate(zs), np.concatenate(ys)


def main():
    train_loader, _, test_loader, _, _, _ = get_loaders(
        batch_size=32
    )

    model = Autoencoder3D(latent_dim=32).to(DEVICE)

    model.load_state_dict(
        torch.load(
            ROOT / "outputs" / "checkpoints" / "ae_best.pt",
            map_location=DEVICE,
        )
    )

    model.eval()

    z_train, y_train = encode_loader(model, train_loader)
    z_test, y_test = encode_loader(model, test_loader)

    print(f"Train latent vectors: {z_train.shape}")
    print(f"Test latent vectors:  {z_test.shape}")
    print()

    results = []
    for i, name in enumerate(PARAM_COLUMNS):
        reg = LinearRegression()

        reg.fit(
            z_train,
            y_train[:, i],
        )

        pred = reg.predict(z_test)

        r2 = r2_score(
            y_test[:, i],
            pred,
        )

        mae = mean_absolute_error(
            y_test[:, i],
            pred,
        )
        
        results.append(
            {
                "parameter": name,
                "r2": r2,
                "mae": mae,
            }
        )

        print(
            f"{name:25s} "
            f"R2={r2:8.4f} "
            f"MAE={mae:8.4f}"
        )

    output_path = (
        ROOT
        / "outputs"
        / "experiments"
        / "latent_probe_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        results
    ).to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved: {output_path}"
    )

if __name__ == "__main__":
    main()