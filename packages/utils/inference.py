from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn
import torch.utils.data


def inference(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    save_folder: Path,
    model_name: str,
    extra: str,
    device: torch.device,
) -> float:
    """Tests model in data from the test_dataloader.

    Evaluates the prediction probabilities for the classes the data are separated to. 
    Saves every image in the save_folder/model_name/extra folder and provides information
    about the classification on the title of the image.

    It also returns the overall accuracy of the model on the test_dataloader.

    Args:
      model: A PyTorch model to be trained and tested.
      test_dataloader: A DataLoader instance for the model to be tested on (NUM_BATCHES = 1).
      class_names: A list of the classes the model is trained on.
      save_folder: A Path instance to save the image.
      model_name: The model's name to use it as a subfolder where the images will be saved.
      extra: An extra string to add to the subfolder where the images will be saved. It provides
        further information about the model and the training process.
      device: A target device to compute on ("cuda", "cpu", "mps").

    Returns:
      results_acc: The overall accuracy of the model on the test_dataloader.
    """
    save_folder = save_folder / model_name / extra
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
