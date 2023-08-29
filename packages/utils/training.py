import os
import torch
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm
from typing import Dict, List, Union
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.datasets
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassFBetaScore

from packages.models.tiny_vgg import TinyVGG
from packages.utils.storage import save_model


def _train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def _test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: Union[torch.utils.tensorboard.writer.SummaryWriter, None] = None
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through _train_step() and _test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer in debug/runs/ folder.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = _train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        test_loss, test_acc = _test_step(model=model,
                                         dataloader=test_dataloader,
                                         loss_fn=loss_fn,
                                         device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "test_loss": test_loss
                },
                global_step=epoch
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "train_acc": train_acc,
                    "test_acc": test_acc
                },
                global_step=epoch
            )

            writer.close()

    return results


def inference(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    save_folder: Path,
    model_name: str,
    device: torch.device,
) -> float:
    """Tests model in data found from the internet and some samples from the dataset.

    Evaluates the prediction probabilities for the classes the data are separated to. 
    Saves every image in the save_folder/model_name folder and provides information
    about the classification on the title of the image.
    It also returns the overall accuracy of the model on the test_dataloader.

    Args:
      model: A PyTorch model to be trained and tested.
      test_dataloader: A DataLoader instance for the model to be tested on (NUM_BATCHES = 1).
      class_names: A list of the classes the model is trained on.
      save_folder: A Path instance to save the image.
      model_name: The model's name to use it as a subfolder where the images will be saved.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      results_acc: The overall accuracy of the model on the test_dataloader.
    """
    save_folder = save_folder / model_name
    save_folder.mkdir(exist_ok=True, parents=True)

    results_acc = 0

    model.eval()
    with torch.inference_mode():
        for i, (imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds_logits = model(imgs)

            preds = torch.softmax(preds_logits, dim=1).max()
            preds_class = preds_logits.argmax(dim=1)

            prob = f"{preds.item():.4f}"
            preds_class = class_names[preds_class.item()]
            truth = class_names[labels.item()]

            plt.imshow(imgs.squeeze().to("cpu").permute(1, 2, 0))
            plt.axis(False)

            title_text = f"Truth: {truth} | Preds: {preds_class} | Prob: {prob}"
            if preds_class == truth:
                results_acc += 1
                plt.title(title_text, color="green")
            else:
                plt.title(title_text, color="red")

            plt.savefig(save_folder / f"{truth}_{preds_class}_{prob}_{i}.png")
            plt.close()

        results_acc = results_acc / len(test_dataloader)

    return results_acc


def _cross_validation_train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> tuple[float, float, float, float, float]:

    model.train()

    train_loss, train_acc = 0, 0

    # Classification metrics
    accuracy_fn = MulticlassAccuracy(num_classes=2).to(device)
    precision_fn = MulticlassPrecision(num_classes=2).to(device)
    recall_fn = MulticlassRecall(num_classes=2).to(device)
    f_score_fn = MulticlassFBetaScore(num_classes=2, beta=2.0).to(device)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = loss_fn(output, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_class = output.argmax(dim=1)

        accuracy_fn.update(pred_class, target)
        precision_fn.update(pred_class, target)
        recall_fn.update(pred_class, target)
        f_score_fn.update(pred_class, target)

    train_loss = train_loss / len(train_loader)

    train_acc = accuracy_fn.compute().item()
    train_pr = precision_fn.compute().item()
    train_rc = recall_fn.compute().item()
    train_fscore = f_score_fn.compute().item()

    return train_loss, train_acc, train_pr, train_rc, train_fscore


def _cross_validation_test(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> tuple[float, float, float, float, float]:

    model.eval()

    test_loss = 0

    # Classification metrics
    accuracy_fn = MulticlassAccuracy(num_classes=2).to(device)
    precision_fn = MulticlassPrecision(num_classes=2).to(device)
    recall_fn = MulticlassRecall(num_classes=2).to(device)
    f_score_fn = MulticlassFBetaScore(num_classes=2, beta=2.0).to(device)

    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()

            pred = output.argmax(dim=1)

            accuracy_fn.update(pred, target)
            precision_fn.update(pred, target)
            recall_fn.update(pred, target)
            f_score_fn.update(pred, target)

    test_loss = test_loss / len(test_loader)

    test_acc = accuracy_fn.compute().item()
    test_pr = precision_fn.compute().item()
    test_rc = recall_fn.compute().item()
    test_fscore = f_score_fn.compute().item()

    return test_loss, test_acc, test_pr, test_rc, test_fscore


def _get_model(
    model_name: str,
    hidden_units: int
) -> torch.nn.Module:
    if model_name == "tiny_vgg":
        model = TinyVGG(
            input_shape=3,
            hidden_units=hidden_units,
            output_shape=2
        )
    else:
        raise ValueError(
            f"Model name {model_name} is not supported. "
            "Please choose between 'tiny_vgg'."
        )

    return model


def _get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float = 1e-3,
) -> torch.optim.Optimizer:
    # Initialize the optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(
            f"Optimizer name {optimizer_name} is not supported. "
            "Please choose between 'adam and SGD'."
        )

    return optimizer


def k_fold_cross_validation(
    model_name: str,
    train_dataset: torchvision.datasets.ImageFolder,
    loss_fn: torch.nn.Module,
    hidden_units: int,
    device: torch.device,
    num_epochs: int,
    root_dir: Path,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",
    num_folds: int = 1,
    save_models: bool = False,
    writer: Union[torch.utils.tensorboard.writer.SummaryWriter, None] = None
):
    # Loss function
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Loop through each fold
    results = {}

    train_indices = np.arange(len(train_dataset))

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_indices)):
        print(f"Fold {fold + 1}")
        print("-------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx.tolist()),
        )
        test_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx.tolist()),
        )

        # Initialize the model
        model = _get_model(
            model_name=model_name,
            hidden_units=hidden_units
        ).to(device)

        # Initialize the optimizer
        optimizer = _get_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
        )

        # Train the model on the current fold
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc, train_pr, train_rc, train_fscore = _cross_validation_train(
                model=model,
                device=device,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer
            )

            print(
                f"Epoch: {epoch+1} | "
                f"Loss: {train_loss:.4f} | "
                f"Accuracy: {train_acc * 100:.2f}% | "
                f"Precession: {train_pr * 100:.2f}% | "
                f"Recall: {train_rc * 100:.2f}% | "
                f"F-Score: {train_fscore * 100:.2f}% | "
            )

            if writer:
                writer.add_scalars(
                    main_tag="train_loss",
                    tag_scalar_dict={
                        f"{fold}_f": train_loss,
                    },
                    global_step=epoch
                )
                writer.add_scalars(
                    main_tag="train_accuracy",
                    tag_scalar_dict={
                        f"{fold}_f": train_acc,
                    },
                    global_step=epoch
                )
                writer.add_scalars(
                    main_tag="train_precession",
                    tag_scalar_dict={
                        f"{fold}_f": train_pr,
                    },
                    global_step=epoch
                )
                writer.add_scalars(
                    main_tag="train_recall",
                    tag_scalar_dict={
                        f"{fold}_f": train_rc,
                    },
                    global_step=epoch
                )
                writer.add_scalars(
                    main_tag="train_fscore",
                    tag_scalar_dict={
                        f"{fold}_f": train_fscore,
                    },
                    global_step=epoch
                )
                writer.close()

        # Save the model for the current fold
        if save_models:
            models_path = root_dir / "models"

            infos = f"{fold}_f_{num_epochs}_e_{batch_size}_bs_{hidden_units}_hu_{learning_rate}_lr"
            model_save_name = f"{model_name}-{infos}.pth"

            save_model(
                model=model,
                MODELS_PATH=models_path,
                MODEL_NAME=model_save_name,
            )

        # Evaluate the model on the test set
        test_loss, test_acc, test_pr, test_rc, test_fscore = _cross_validation_test(
            model=model,
            device=device,
            test_loader=test_loader,
            loss_fn=loss_fn,
        )

        results[fold] = {
            "Accuracy": test_acc,
            "Precession": test_pr,
            "Recall": test_rc,
            "F-Score": test_fscore
        }

        # Print the results for the current fold
        print(
            "Validation Results: "
            f"Loss: {test_loss:.4f} | "
            f"Accuracy: {test_acc * 100:.2f}% | "
            f"Precession: {test_pr * 100:.2f}% | "
            f"Recall: {test_rc * 100:.2f}% | "
            f"F-Score: {test_fscore * 100:.2f}% |\n"
        )

        if writer:
            writer.add_scalars(
                main_tag="test_loss",
                tag_scalar_dict={
                    f"{num_folds}_f_total": test_loss,
                },
                global_step=fold
            )
            writer.add_scalars(
                main_tag="test_accuracy",
                tag_scalar_dict={
                    f"{num_folds}_f_total": test_acc,
                },
                global_step=fold
            )
            writer.add_scalars(
                main_tag="test_precession",
                tag_scalar_dict={
                    f"{num_folds}_f_total": test_pr,
                },
                global_step=fold
            )
            writer.add_scalars(
                main_tag="test_recall",
                tag_scalar_dict={
                    f"{num_folds}_f_total": test_rc,
                },
                global_step=fold
            )
            writer.add_scalars(
                main_tag="test_fscore",
                tag_scalar_dict={
                    f"{num_folds}_f_total": test_fscore,
                },
                global_step=fold
            )
            writer.close()

    # Print fold results
    print(f'K-Fold Cross Validation Results for {num_folds} Folds')
    print('--------------------------------------------')

    metrics_sum = {
        "Accuracy": 0.0,
        "Precession": 0.0,
        "Recall": 0.0,
        "F-Score": 0.0
    }

    for key, metrics_dict in results.items():
        print(f"Fold {key + 1}: ", end="")
        for key, metric in metrics_dict.items():
            print(f"{key}: {metric * 100:.2f}% | ", end="")
            metrics_sum[key] += metric
        print()

    print("\nAverage: ", end="")

    for key, metric_sum in metrics_sum.items():
        metric_sum = metric_sum / num_folds
        print(f"{key}: {metric_sum * 100:.2f}% | ", end="")
