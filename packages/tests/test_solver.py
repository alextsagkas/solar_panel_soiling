from pathlib import Path
from typing import Dict

import torch
import torch.utils.data
from torchvision.datasets import ImageFolder

from packages.utils.configuration import GetModel
from packages.utils.solver import Solver
from packages.utils.transforms import GetTransforms


def test_solver(
    model_obj: GetModel,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    train_dir: Path,
    test_dir: Path,
    transform_obj: GetTransforms,
    root_dir: Path,
) -> None:

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=transform_obj.get_train_transform(),
    )
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=transform_obj.get_test_transform()
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    solver = Solver(
        model=model_obj.get_model(),
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform_name=transform_obj.transform_name,
        root_dir=root_dir,
    )

    solver.train_model()
