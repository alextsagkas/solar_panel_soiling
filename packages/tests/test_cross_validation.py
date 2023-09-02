from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.backends.mps
import torch.utils.tensorboard.summary
from torchvision import datasets

from packages.utils.configuration import GetModel
from packages.utils.k_fold_cross_val import k_fold_cross_validation
from packages.utils.tensorboard import create_writer
from packages.utils.transforms import GetTransforms


def test_cross_validation(
    model_obj: GetModel,
    models_path: Path,
    train_dir: Path,
    test_dir: Path,
    num_folds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    experiment_name: str,
    transform_obj: GetTransforms,
) -> Tuple[Dict[str, float], str]:
    """Trains and tests a model using k-fold cross validation.

    Args:
        model_obj (GetModel): The model object to use for the training.
        models_path (Path): The directory where the models will be saved.
        train_dir (Path): The directory where the training data are located.
        test_dir (Path): The directory where the testing data are located.
        num_folds (int): The number of folds to use in k-fold cross validation (>=1).
        num_epochs (int): The number of epochs to train for, in each fold (>=0). 
        batch_size (int): The number of samples per batch (power of 2).
        learning_rate (float): The learning rate (>=0 and <=1).
        optimizer_name (str): The optimizer's name to pick the optimizer.
        experiment_name (str): The experiment's name to use it as a subfolder where the images will
        transform_obj (GetTransforms): The transform object to use for the data. It has the
            functionality to return the train and test transforms based on its initialization.

    Returns:
        Tuple[Dict[str, float], str]: Returns a dictionary with the classification metrics of the 
            test and a string with the extra information concerning the training.
    """
    # Instantiate the writer
    EXTRA = f"{num_folds}_f_{num_epochs}_e_{batch_size}_bs_{model_obj.hidden_units}_hu_{learning_rate}_lr"

    writer = create_writer(
        experiment_name=experiment_name,
        model_name=model_obj.model_name,
        transform_name=transform_obj.transform_name,
        extra=EXTRA,
    )

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train and Test Datasets
    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=transform_obj.get_train_transform(),
        target_transform=None
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=transform_obj.get_test_transform(),
        target_transform=None
    )

    # Train and evaluate the model
    metrics_avg = k_fold_cross_validation(
        model_obj=model_obj,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        models_path=models_path,
        experiment_name=experiment_name,
        transform_name=transform_obj.transform_name,
        optimizer_name=optimizer_name,
        num_folds=num_folds,
        writer=writer
    )

    return metrics_avg, EXTRA
