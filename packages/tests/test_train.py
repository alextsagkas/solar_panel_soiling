from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.backends.mps

from packages.utils.configuration import GetModel, _get_optimizer
from packages.utils.load_data import get_dataloaders
from packages.utils.storage import save_model
from packages.utils.tensorboard import create_writer
from packages.utils.training import train
from packages.utils.transforms import GetTransforms


def test_train(
    model_obj: GetModel,
    train_dir: Path,
    test_dir: Path,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    optimizer_name: str,
    learning_rate: float,
    models_path: Path,
    experiment_name: str,
    transform_obj: GetTransforms,
) -> Tuple[Dict[str, float], str]:
    # TODO: Add docstring
    # Get the dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        train_dir=str(train_dir),
        train_transform=transform_obj.get_train_transform(),
        test_dir=str(test_dir),
        test_transform=transform_obj.get_test_transform(),
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Instantiate the model
    model = model_obj.get_model()

    # Instantiate the writer
    EXTRA = f"{num_epochs}_e_{batch_size}_bs_{model_obj.hidden_units}_hu_{learning_rate}_lr"

    writer = create_writer(
        experiment_name=experiment_name,
        model_name=model_obj.model_name,
        transform_name=transform_obj.transform_name,
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
        device=model_obj.device,
        writer=writer
    )

    # Save the model
    save_model(
        model=model,
        models_path=models_path,
        model_name=model_obj.model_name,
        experiment_name=experiment_name,
        transform_name=transform_obj.transform_name,
        extra=EXTRA
    )

    return metrics, EXTRA
