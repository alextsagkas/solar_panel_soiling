from pathlib import Path

import torch
import torch.utils.data
from torchvision.datasets import ImageFolder

from packages.utils.models import GetModel
from packages.utils.solver import KfoldSolver, Solver
from packages.utils.transforms import GetTransforms


def test_solver(
    model_obj: GetModel,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
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
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform_name=transform_obj.transform_name,
        root_dir=root_dir,
    )

    solver.train_model()


def test_kfold_solver(
    model_obj: GetModel,
    device: torch.device,
    num_folds: int,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
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

    solver = KfoldSolver(
        model_obj=model_obj,
        device=device,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_name=optimizer_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform_name=transform_obj.transform_name,
        root_dir=root_dir,
    )

    solver.train_model()
