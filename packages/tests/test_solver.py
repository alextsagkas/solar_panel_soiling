from typing import List

import torch
import torch.utils.data
from torchvision.datasets import ImageFolder

from packages.utils.configuration import test_dir, train_dir
from packages.utils.models import GetModel
from packages.utils.solver import Solver
from packages.utils.transforms import GetTransforms


def test_solver(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
    transform_name: str,
    timestamp_list: List[str],
    device: torch.device,
) -> None:
    """Train and test a model using simple train from Solver.

    Args:
        model_name (str): String that identifies the model to be used.
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The size of the batches to be used in the training and testing.
        optimizer_name (str): String that identifies the optimizer to be used.
        transform_name (str): String that identifies the transform to be used.
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS).
        device (torch.device): A target device to compute on ("cuda", "cpu", "mps").
    """

    transform_obj = GetTransforms(transform_name=transform_name)

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=transform_obj.get_train_transform(),
    )
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=transform_obj.get_test_transform()
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model_obj = GetModel(model_name=model_name)

    solver = Solver(
        model_obj=model_obj,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_name=optimizer_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        timestamp_list=timestamp_list,
    )

    solver.train_model()


def test_kfold_solver(
    model_name: str,
    num_folds: int,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
    transform_name: str,
    timestamp_list: List[str],
    device: torch.device,
) -> None:
    """Train and test a model using k-fold train from Solver.

    Args:
        model_name (str): String that identifies the model to be used.
        num_folds (int): Number of fold to be used in the k-fold training.
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The size of the batches to be used in the training and testing.
        optimizer_name (str): String that identifies the optimizer to be used.
        transform_name (str): String that identifies the transform to be used.
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS).
        device (torch.device): A target device to compute on ("cuda", "cpu", "mps").
    """

    transform_obj = GetTransforms(transform_name=transform_name)

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=transform_obj.get_train_transform(),
    )
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=transform_obj.get_test_transform()
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model_obj = GetModel(model_name=model_name)

    solver = Solver(
        model_obj=model_obj,
        device=device,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_name=optimizer_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        timestamp_list=timestamp_list,
    )

    solver.train_model_kfold()
