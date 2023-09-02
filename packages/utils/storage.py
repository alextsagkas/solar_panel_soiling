from typing import Dict, List, Union

import torch

from packages.utils.configuration import metrics_dir, models_dir
from packages.utils.time import get_time


def save_model(
    model: torch.nn.Module,
    timestamp_list: List[str],
    extra: Union[str, None] = None,
) -> None:
    """Saves the state dict of model in models_dir/YYYY-MM-DD/HH-MM-SS_{extra}.pth.

    Args:
        model (torch.nn.Module): Model to save.
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS).
        extra (str): Extra information concerning the training (used as name of file saved).
    """
    model_save_dir = models_dir / timestamp_list[0]
    model_save_dir.mkdir(exist_ok=True, parents=True)

    if extra is None:
        model_save_path = model_save_dir / f"{timestamp_list[1]}.pth"
    else:
        model_save_path = model_save_dir / f"{timestamp_list[1]}_{extra}.pth"

    print(f"[INFO] Saving model to: {model_save_path}")

    torch.save(
        obj=model.state_dict(),
        f=model_save_path
    )


def save_metrics(
    metrics: Dict[str, float],
    timestamp_list: List[str],
) -> None:
    """Saves the metrics of the experiment in metrics_dir/YYYY-MM-DD/HH-MM-SS.pth.


    Args:
        metrics (Dict[str, float]): The metrics of the experiment (classification metrics, loss,
            duration).
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS).
    """
    save_metrics_dir = metrics_dir / timestamp_list[0]
    save_metrics_dir.mkdir(exist_ok=True, parents=True)

    metrics_file = save_metrics_dir / f"{timestamp_list[1]}.txt"

    print(f"[INFO] Saving metrics to: {metrics_file}")

    with open(metrics_file, "w") as f:
        for key, metric in metrics.items():
            if key == "time":
                f.write(f"{key}: {get_time(metric)}\n")
            elif key == "loss":
                f.write(f"{key}: {metric:.4f}\n")
            else:
                f.write(f"{key}: {metric * 100:.2f}%\n")
