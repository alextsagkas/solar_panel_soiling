from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.backends.mps

from packages.utils.configuration import _get_model, _get_optimizer
from packages.utils.load_data import get_dataloaders
from packages.utils.storage import save_model
from packages.utils.tensorboard import create_writer
from packages.utils.training import train
from packages.utils.transforms import test_data_transform, train_data_transform


def test_train(
    model_name: str,
    train_dir: Path,
    test_dir: Path,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    hidden_units: int,
    optimizer_name: str,
    learning_rate: float,
    device: torch.device,
    models_path: Path
) -> Tuple[Dict[str, float], str, str]:
    EXPERIMENT_NAME = "test_train"

    # Get the dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        train_dir=str(train_dir),
        train_transform=train_data_transform,
        test_dir=str(test_dir),
        test_transform=test_data_transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Instantiate the model
    model = _get_model(
        model_name=model_name,
        hidden_units=hidden_units
    ).to(device)

    # Instantiate the writer
    EXTRA = f"{num_epochs}_e_{batch_size}_bs_{hidden_units}_hu_{learning_rate}_lr"

    writer = create_writer(
        experiment_name=EXPERIMENT_NAME,
        model_name=model_name,
        extra=EXTRA
    )

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = _get_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        learning_rate=learning_rate
    )

    # Train the model
    metrics = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=num_epochs,
        device=device,
        writer=writer
    )

    # Save the model
    save_model(
        model=model,
        models_path=models_path,
        model_name=model_name,
        experiment_name=EXPERIMENT_NAME,
        extra=EXTRA
    )

    return metrics, EXTRA, EXPERIMENT_NAME
