from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch.nn
import torch.utils.data
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)


def inference(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    test_model_path: Path,
    model_name: str,
    experiment_name: str,
    extra: str,
    device: torch.device,
) -> Dict[str, float]:
    """Tests model in data from the test_dataloader.

    Evaluates the prediction probabilities for the classes the data are separated to. 
    Saves every image in the test_model_path/model_name/experiment_name/extra/ folder and
    provides information about the classification on the title of the image.

    It also returns a dictionary containing the classification metrics (accuracy, precession, recall, f-beta score).

    Args:
      model (torch.nn.Module): A PyTorch model to be trained and tested.
      test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be 
        tested on (using size of batches = 1).
      class_names (List[int]): A list of the classes the model is trained on.
      test_model_path: (Path): The directory where the images will be saved.
      model_name (str): The model's name to use it as a subfolder where the images will be saved.
      experiment_name (str): The experiment's name to use it as a subfolder where the images will 
        be saved.
      extra (str): A string used as a subfolder where the images will be saved. It 
        provides further information about the model and the training process.
      device (torch.device): A target device to compute on ("cuda", "cpu", "mps").

    Returns:
      Dict[str, float]: A dictionary containing the classification metrics (accuracy, precession, recall, f-beta score).
    """
    save_folder = test_model_path / model_name / experiment_name / extra
    save_folder.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Saving images in {save_folder}")

    model.eval()

    results_metrics = {
        "accuracy": MulticlassAccuracy(num_classes=2).to(device),
        "precession": MulticlassPrecision(num_classes=2).to(device),
        "recall": MulticlassRecall(num_classes=2).to(device),
        "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(device),
    }

    with torch.inference_mode():
        for i, (imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds_logits = model(imgs)

            preds = torch.softmax(preds_logits, dim=1).max()
            preds_class = preds_logits.argmax(dim=1)

            for key, _ in results_metrics.items():
                results_metrics[key].update(preds_class, labels)

            prob = f"{preds.item():.4f}"
            preds_class = class_names[preds_class.item()]
            truth = class_names[labels.item()]

            plt.imshow(imgs.squeeze().to("cpu").permute(1, 2, 0))
            plt.axis(False)

            title_text = f"Truth: {truth} | Preds: {preds_class} | Prob: {prob}"
            if preds_class == truth:
                plt.title(title_text, color="green")
            else:
                plt.title(title_text, color="red")

            plt.savefig(save_folder / f"{truth}_{preds_class}_{prob}_{i}.png")
            plt.close()

    for key, _ in results_metrics.items():
        results_metrics[key] = results_metrics[key].compute().item()

    return results_metrics
