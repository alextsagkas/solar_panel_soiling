from typing import Dict, List

import torch
import torch.utils.data
from torchvision.datasets import ImageFolder

from packages.utils.configuration import test_dir, train_dir
from packages.utils.models import GetModel
from packages.utils.solver import Solver
from packages.utils.transforms import GetTransforms


def test_resume(
    load_config: Dict,
    num_epochs: int,
    batch_size: int,
    optimizer_name: str,
    train_transform_name: str,
    test_transform_name: str,
    timestamp_list: List[str],
    device: torch.device,
    **kwargs,
) -> None:
    """Resume training and testing of a loaded model using simple train from Solver.

    Via the kwargs the training can be further customized. An example of the supported 
    kwargs is the following:
        kwargs = {
            "optimizer_config": {"weight_decay": 0.001},
            "train_transform_config": {"random_horizontal_flip": 0.45},
            "test_transform_config": {"random_rotation": 180},
        }

    Args:
        load_config (Dict): Dictionary with the configuration of the loaded model. An example of a
            load_config is the following:
                load_config = {
                    "checkpoint_timestamp_list": ["2021-08-01", "2021-08-01_12-00-00"],
                    "load_epoch": 1,
                }
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The size of the batches to be used in the training and testing.
        optimizer_name (str): String that identifies the optimizer to be used.
        train_transform_name (str): String that identifies the transform to be used on the 
            train data.
        test_transform_name (str): String that identifies the transform to be used on the
            test data.
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS).
        device (torch.device): A target device to compute on ("cuda", "cpu", "mps").
        kwargs (dict): Dictionary of optional arguments. Defaults to None.
    """
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
        load_checkpoint=True,
        load_config=load_config,
    )

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
        optimizer_config=optimizer_config,
    )

    solver.train_model()
