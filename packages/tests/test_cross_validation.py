from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.backends.mps
import torch.utils.tensorboard.summary
from torchvision import datasets

from packages.utils.k_fold_cross_val import k_fold_cross_validation
from packages.utils.tensorboard import create_writer
from packages.utils.transforms import test_data_transform, train_data_transform


def test_cross_validation(
    root_dir: Path,
    models_path: Path,
    train_dir: Path,
    test_dir: Path,
    device: torch.device,
    num_folds: int,
    num_epochs: int,
    batch_size: int,
    hidden_units: int,
    learning_rate: float,
    model_name: str,
    optimizer_name: str,
    save_models: bool,
) -> Tuple[Dict[str, float], str, str]:
    # Instantiate the writer
    EXPERIMENT_NAME = "test_kfold"
    EXTRA = f"{num_folds}_f_{num_epochs}_e_{batch_size}_bs_{hidden_units}_hu_{learning_rate}_lr"

    writer = create_writer(
        experiment_name=EXPERIMENT_NAME,
        model_name=model_name,
        extra=EXTRA
    )

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train and Test Datasets
    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_data_transform,
        target_transform=None
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=test_data_transform,
        target_transform=None
    )

    # Train and evaluate the model
    metrics_avg = k_fold_cross_validation(
        model_name=model_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        hidden_units=hidden_units,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        models_path=models_path,
        experiment_name=EXPERIMENT_NAME,
        optimizer_name=optimizer_name,
        num_folds=num_folds,
        save_models=save_models,
        writer=writer
    )

    return metrics_avg, EXTRA, EXPERIMENT_NAME
