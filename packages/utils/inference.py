from typing import List

import matplotlib.pyplot as plt
import torch.nn
import torch.utils.data
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)

from packages.utils.configuration import test_model_dir
from packages.utils.storage import save_metrics


def inference(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    timestamp_list: List[str],
    save_images: bool = False,
) -> None:
    """Tests model in data from the test_dataloader.

    Evaluates the prediction probabilities for the classes the data are separated to. Saves a dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and prints it on the console.

    It also optionally saves every image in the test_model_dir/timestamp_list[0]/timestamp_list[1] folder and provides information about the classification on the title of the image. This behavior is controlled by the save_images argument.

    **Args:**
    
      model : torch.nn.Module 
		A PyTorch model to be trained and tested.
      test_dataloader : torch.utils.data.DataLoader 
		A DataLoader instance for the model to be tested on (using size of batches = 1).
      class_names : List[int] 
		A list of the classes the model is trained on.
      device : torch.device 
		A target device to compute on ("cuda", "cpu", "mps").
      timestamp_list : List[str] 
		List of timestamp (YYYY-MM-DD, HH-MM-SS).
    """
    if save_images:
        save_folder = test_model_dir / timestamp_list[0] / timestamp_list[1]
        save_folder.mkdir(exist_ok=True, parents=True)
        print(f"[INFO] Saving images in {save_folder}")
    else:
        print("[INFO] Not saving images.")

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

            preds_class = preds_logits.argmax(dim=1)

            for key, _ in results_metrics.items():
                results_metrics[key].update(preds_class, labels)

            preds_class = class_names[preds_class.item()]

            if save_images:
                preds = torch.softmax(preds_logits, dim=1).max()
                prob = f"{preds.item():.4f}"
                truth = class_names[labels.item()]

                plt.imshow(imgs.squeeze().to("cpu").permute(1, 2, 0))
                plt.axis(False)

                title_text = f"Truth: {truth} | Preds: {preds_class} | Prob: {prob}"
                if preds_class == truth:
                    plt.title(title_text, color="green")
                else:
                    plt.title(title_text, color="red")

                plt.savefig(save_folder / f"{truth}_{preds_class}_{prob}_{i}.png")  # type: ignore
                plt.close()

    print("result || ", end="")

    for key, _ in results_metrics.items():
        results_metrics[key] = results_metrics[key].compute().item()
        print(f"{key}: {results_metrics[key]*100:.2f}% | ", end="")
    print()

    save_metrics(
        metrics=results_metrics,
        timestamp_list=timestamp_list,
    )
