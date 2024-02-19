from pathlib import Path
from typing import List, Union

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
    train_transform_name: str,
    test_transform_name: str,
    timestamp_list: List[str],
    device: torch.device,
    train_dir: Path = train_dir,
    test_dir: Path = test_dir,
    scheduler_name: Union[str, None] = None,
    **kwargs,
) -> None:
    """Train and test a model using simple train from Solver.

    Via the kwargs the training can be further customized. An example of the supported kwargs is the following::

        kwargs = {
            "model_config": {"hidden_units": 128},
            "optimizer_config": {"weight_decay": 0.001},
            "train_transform_config": {"random_horizontal_flip": 0.45},
            "test_transform_config": {"random_rotation": 180},
            "scheduler_config": {"step_size": 10, "gamma": 0.1},
        }

    **Args:**

        model_name : str 
			String that identifies the model to be used.
        num_epochs : int 
			The number of epochs to train the model.
        batch_size : int 
			The size of the batches to be used in the training and testing.
        optimizer_name : str 
			String that identifies the optimizer to be used.
        train_transform_name : str 
			String that identifies the transform to be used on the train data.
        test_transform_name : str 
			String that identifies the transform to be used on the test data.
        timestamp_list : List[str] 
			List of timestamp (YYYY-MM-DD, HH-MM-SS).
        device : torch.device 
			A target device to compute on ("cuda", "cpu", "mps").
        train_dir : Path, optional 
			Path to the directory containing the training data. Defaults to train_dir.
        test_dir : Path, optional 
			Path to the directory containing the testing data. Defaults to test_dir.
        scheduler_name : Union[str, None], optional 
			String that identifies the scheduler to be used. Defaults to None.
        kwargs : dict 
			Dictionary of optional arguments. Defaults to None.
    """
    model_config = kwargs.get("model_config", None)
    optimizer_config = kwargs.get("optimizer_config", None)
    scheduler_config = kwargs.get("scheduler_config", None)
    train_transform_config = kwargs.get("train_transform_config", None)
    test_transform_config = kwargs.get("test_transform_config", None)

    transform_obj = GetTransforms(
        train_transform_name=train_transform_name,
        train_config=train_transform_config,
        test_transform_name=test_transform_name,
        test_config=test_transform_config,
    )

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=transform_obj.get_train_transform(),
    )
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=transform_obj.get_test_transform()
    )

    print(
        f"[INFO] Using {train_dir} data to train the model and "
        f"{test_dir} data to test the model."
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model_obj = GetModel(
        model_name=model_name,
        config=model_config,
    )

    solver = Solver(
        model_obj=model_obj,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        timestamp_list=timestamp_list,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )

    solver.train_model()


def test_kfold_solver(
    model_name: str,
    num_folds: int,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
    train_transform_name: str,
    test_transform_name: str,
    timestamp_list: List[str],
    device: torch.device,
    **kwargs,
) -> None:
    """Train and test a model using k-fold train from Solver.

    Via the kwargs the training can be further customized. An example of the supported
    kwargs is the following::

        kwargs = {
            "model_config": {"hidden_units": 128},
            "optimizer_config": {"weight_decay": 0.001},
            "train_transform_config": {"random_horizontal_flip": 0.45},
            "test_transform_config": {"random_rotation": 180},
        }

    **Args:**
        model_name : str 
			String that identifies the model to be used.
        num_folds : int 
			Number of fold to be used in the k-fold training.
        num_epochs : int 
			The number of epochs to train the model.
        batch_size : int 
			The size of the batches to be used in the training and testing.
        optimizer_name : str 
			String that identifies the optimizer to be used.
        train_transform_name : str 
			String that identifies the transform to be used on the train data.
        test_transform_name : str 
			String that identifies the transform to be used on the test data.
        timestamp_list : List[str] 
			List of timestamp (YYYY-MM-DD, HH-MM-SS).
        device : torch.device 
			A target device to compute on ("cuda", "cpu", "mps").
        kwargs : dict 
			Dictionary of optional arguments. Defaults to None.
    """
    model_config = kwargs.get("model_config", None)
    optimizer_config = kwargs.get("optimizer_config", None)
    train_transform_config = kwargs.get("train_transform_config", None)
    test_transform_config = kwargs.get("test_transform_config", None)

    transform_obj = GetTransforms(
        train_transform_name=train_transform_name,
        train_config=train_transform_config,
        test_transform_name=test_transform_name,
        test_config=test_transform_config,
    )

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=transform_obj.get_train_transform(),
    )
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=transform_obj.get_test_transform()
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model_obj = GetModel(
        model_name=model_name,
        config=model_config,
    )

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
        optimizer_config=optimizer_config,
    )

    solver.train_model_kfold()
