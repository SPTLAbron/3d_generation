from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import (
    DATA_ROOT,
    PARAM_COLUMNS,
    RANDOM_SEED,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


class TrophyDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT):
        self.root = Path(data_root)

        metadata_path = self.root / "metadata" / "parameters.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        self.df = pd.read_csv(metadata_path)

        missing_columns = [
            column
            for column in ["filename", *PARAM_COLUMNS]
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Metadata is missing required columns: {missing_columns}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        voxel_filename = row["filename"].replace(".obj", ".npy")
        voxel_path = self.root / "voxels" / voxel_filename

        if not voxel_path.exists():
            raise FileNotFoundError(
                f"Voxel file not found: {voxel_path}"
            )

        voxel = np.load(voxel_path)

        if voxel.shape != (32, 32, 32):
            raise ValueError(
                f"Expected voxel shape (32, 32, 32), "
                f"got {voxel.shape} for {voxel_path}"
            )

        shape = torch.from_numpy(voxel).float().unsqueeze(0)

        params = torch.tensor(
            row[PARAM_COLUMNS].values.astype(np.float32),
            dtype=torch.float32,
        )

        return shape, params


def get_datasets(seed=RANDOM_SEED):
    dataset = TrophyDataset()

    n_total = len(dataset)

    train_n = int(TRAIN_FRACTION * n_total)
    val_n = int(VAL_FRACTION * n_total)
    test_n = n_total - train_n - val_n

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_n, val_n, test_n],
        generator=generator,
    )

    return train_dataset, val_dataset, test_dataset


def get_loaders(batch_size=16, seed=RANDOM_SEED, num_workers=0):
    train_dataset, val_dataset, test_dataset = get_datasets(seed)

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":
    dataset = TrophyDataset()

    shape, params = dataset[0]

    print(f"Dataset size: {len(dataset)}")
    print(f"Shape tensor: {shape.shape}")
    print(f"Parameter tensor: {params.shape}")

    train_dataset, val_dataset, test_dataset = get_datasets()

    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")