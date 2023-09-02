import os
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict

import torch
import torch.utils.data
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm
from typing_extensions import Self

from packages.utils.configuration import GetOptimizer
from packages.utils.storage import save_model, save_results


class Solver:
    """Class that trains and tests a Pytorch model

    Args:
        model (torch.nn.Module): The model to be trained and tested.
        device (torch.device): Device to be used to load the model.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per batch.
        loss_fn (torch.nn.Module): Loss function to be used.
        optimizer_name (str): String that identifies the optimizer to be used.
        learning_rate (float): Learning rate to be used by the optimizer.
        train_dataset (torch.utils.data.Dataset): Train dataset.
        test_dataset (torch.utils.data.Dataset): Test dataset.
        transform_name (str): String that identifies the transform to be used.
        root_dir (Path): Path to the root directory.

    Attributes:
        experiment_name (str): Name of the experiment ("test_train").
        model (torch.nn.Module): The model to be trained and tested.
        model_name (str): String corresponds to the model's class name (used for storage 
            path identification).
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per batch.
        loss_fn (torch.nn.Module): Loss function to be used.
        optimizer_name (str): String that identifies the optimizer to be used.
        learning_rate (float): Learning rate to be used by the optimizer.
        device (torch.device): Device to be used to load the model.
        root_dir (Path): Path to the root directory.
        models_dir (Path): Path to the models directory.
        test_model_dir (Path): Path to the test model directory.
        train_dataset (torch.utils.data.Dataset): Train dataset.
        test_dataset (torch.utils.data.Dataset): Test dataset.
        transform_name (str): String that identifies the transform to be used.
        extra (str): String that contains the number of epochs, batch size and learning rate (used
            for storage path identification).

    Methods:
        _train_step: Trains a PyTorch model for one epoch.
        _test_step: Tests a PyTorch model for one epoch.
        _create_writer: Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to
            a specific log_dir.
        _display_metrics: Receives data about the phase and epoch and prints out the metrics while
            saving them to a tensorboard writer.
        train_model: Trains and tests a PyTorch model.
    """

    def __init__(
        self: Self,
        model: torch.nn.Module,
        device: torch.device,
        num_epochs: int,
        batch_size: int,
        loss_fn: torch.nn.Module,
        optimizer_name: str,
        learning_rate: float,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        transform_name: str,
        root_dir: Path,
    ) -> None:
        """Initializes the Solver class.

        Args:
            model (torch.nn.Module): The model to be trained and tested.
            device (torch.device): Device to be used to load the model.
            num_epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per batch.
            loss_fn (torch.nn.Module): Loss function to be used.
            optimizer_name (str): String that identifies the optimizer to be used.
            learning_rate (float): Learning rate to be used by the optimizer.
            train_dataset (torch.utils.data.Dataset): Train dataset.
            test_dataset (torch.utils.data.Dataset): Test dataset.
            transform_name (str): String that identifies the transform to be used.
            root_dir (Path): Path to the root directory.
        """
        self.experiment_name = "test_train"
        self.model = model
        # Get the model name from the model's class
        self.model_name = str(self.model.__class__).split(".")[-1].replace("'>", "").lower()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.device = device
        # Paths
        self.root_dir = root_dir
        self.models_dir = self.root_dir / "models"
        self.test_model_dir = self.models_dir / "test_model"
        # Datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.transform_name = transform_name
        # Extra
        self.extra = f"{self.num_epochs}_e_{self.batch_size}_bs_{self.learning_rate}_lr"

        print(f"[INFO] Using {self.model_name} model.")

    def _train_step(
        self: Self,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Trains a PyTorch model for one epoch.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to be used.
            dataloader (torch.utils.data.DataLoader): Train DataLoader.

        Returns:
            Dict[str, float]: Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
                metrics = {
                    "accuracy": 0.94,
                    "precession": 0.80,
                    "recall": 0.69,
                    "fscore": 0.76,
                    "loss": 0.18,
                }
        """
        self.model.train()

        train_loss = 0
        train_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=2).to(self.device),
            "precession": MulticlassPrecision(num_classes=2).to(self.device),
            "recall": MulticlassRecall(num_classes=2).to(self.device),
            "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(self.device),
        }

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = y_pred.argmax(dim=1)

            for key, _ in train_metrics.items():
                train_metrics[key].update(y_pred_class, y)

        train_loss = train_loss / len(dataloader)

        for key, _ in train_metrics.items():
            train_metrics[key] = train_metrics[key].compute().item()
        train_metrics["loss"] = train_loss

        return train_metrics

    def _test_step(
        self: Self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Tests a PyTorch model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): Test DataLoader.

        Returns:
            Dict[str, float]: Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
                metrics = {
                    "accuracy": 0.94,
                    "precession": 0.80,
                    "recall": 0.69,
                    "fscore": 0.76,
                    "loss": 0.18,
                }
        """
        self.model.eval()

        test_loss = 0
        test_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=2).to(self.device),
            "precession": MulticlassPrecision(num_classes=2).to(self.device),
            "recall": MulticlassRecall(num_classes=2).to(self.device),
            "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(self.device),
        }

        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                test_pred_logits = self.model(X)

                test_loss += self.loss_fn(test_pred_logits, y)

                test_pred_labels = test_pred_logits.argmax(dim=1)

                for key, _ in test_metrics.items():
                    test_metrics[key].update(test_pred_labels, y)

        test_loss = test_loss / len(dataloader)

        for key, _ in test_metrics.items():
            test_metrics[key] = test_metrics[key].compute().item()
        test_metrics["loss"] = test_loss

        return test_metrics

    def _create_writer(
        self: Self,
    ) -> torch.utils.tensorboard.writer.SummaryWriter:
        """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

        log_dir is a combination of runs/timestamp/model_nam/experiment_name/transform_name/extra.

        Where timestamp is the current date in YYYY-MM-DD format.

        Returns:
            torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
        """

        path = self.root_dir / "debug" / "runs"
        os.makedirs(path, exist_ok=True)

        # Get timestamp of current date (all experiments on certain day live in same folder)
        timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format

        log_dir = os.path.join(
            path,
            timestamp,
            self.model_name,
            self.experiment_name,
            self.transform_name,
            self.extra
        )

        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")

        return SummaryWriter(log_dir=log_dir)

    def _display_metrics(
        self: Self,
        phase: str,
        epoch: int,
        metrics: Dict[str, float],
        writer: torch.utils.tensorboard.writer.SummaryWriter,
    ) -> None:
        """Receives data about the phase and epoch and prints out the metrics associated with them. 
        It also saves the metrics' evolution throughout training/testing to a tensorboard writer.

        Args:
            phase (str): The current phase of "train" or "test".
            epoch (int): The current epoch on "train" phase or total number of epochs on "test" phase.
            metrics (Dict[str, float]): Dictionary containing the classification metrics and loss.
            writer (torch.utils.tensorboard.writer.SummaryWriter):  
                Tensorboard SummaryWriter object. If it is None then metrics will not be saved 
                to tensorboard.

        Raises:
            ValueError: If the phase is not one of "train" or "test".
        """
        supported_phases = ["train", "test"]
        if phase not in supported_phases:
            raise ValueError(f"Phase {phase} not supported. Please choose between {supported_phases}")

        # Print Metrics
        print(f"{phase} || epoch: {epoch} | ", end="")

        for key, metric in metrics.items():
            print(f"{key}: {metric:.4f} | ", end="")
        print()

        # Save Metrics to Tensorboard
        for key, metric in metrics.items():
            writer.add_scalars(
                main_tag=f"{phase}",
                tag_scalar_dict={
                    f"{key}": metric,
                },
                global_step=epoch
            )
        writer.close()

    def train_model(
        self: Self,
    ) -> Dict[str, float]:
        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through _train_step() function for a number of epochs.
        When the training is done, passes the same updated model through _test_step() function,
        to evaluate the model's performance on the test dataset.

        Calculates, prints and stores evaluation metrics on training set throughout.

        Stores metrics to specified writer in debug/runs/folder.

        Returns:
            Dict[str, float]: A dictionary containing the classification metrics, the loss for
            the test dataset, and the time taken to train it. Example:
                metrics = {
                    "accuracy": 0.94,
                    "precession": 0.80,
                    "recall": 0.69,
                    "fscore": 0.76,
                    "loss": 0.18,
                    "time": 0.18,
                }
        """
        # Count model time
        start_time = timer()

        # Dataloaders
        NUM_CORES = os.cpu_count()

        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_CORES if NUM_CORES is not None else 1,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_CORES if NUM_CORES is not None else 1,
        )

        # Create writer
        writer = self._create_writer()

        # Create Optimizer
        optimizer = GetOptimizer(
            optimizer_name=self.optimizer_name,
            model=self.model,
            learning_rate=self.learning_rate,
        ).get_optimizer()

        for epoch in tqdm(range(self.num_epochs)):
            train_metrics = self._train_step(
                dataloader=train_dataloader,
                optimizer=optimizer,
            )

            self._display_metrics(
                phase="train",
                epoch=epoch,
                metrics=train_metrics,
                writer=writer
            )

        save_model(
            model=self.model,
            models_path=self.models_dir,
            model_name=self.model_name,
            experiment_name=self.experiment_name,
            transform_name=self.transform_name,
            extra=self.extra,
        )

        test_metrics = self._test_step(
            dataloader=test_dataloader,
        )
        self._display_metrics(
            phase="test",
            epoch=self.num_epochs,
            metrics=test_metrics,
            writer=writer
        )

        end_time = timer()

        metrics = test_metrics
        metrics["time"] = end_time - start_time

        save_results(
            root_dir=self.root_dir,
            models_name=self.model_name,
            experiment_name=self.experiment_name,
            transform_name=self.transform_name,
            extra=self.extra,
            metrics=metrics,
        )

        return metrics
