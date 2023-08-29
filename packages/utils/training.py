import os
import torch
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm
from typing import Dict, List, Union
import matplotlib.pyplot as plt
from pathlib import Path


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
