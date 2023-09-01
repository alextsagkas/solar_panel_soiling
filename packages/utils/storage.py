from pathlib import Path
from typing import Dict

import torch

from packages.utils.time import get_time


def save_model(
    model: torch.nn.Module,
    models_path: Path,
    model_name: str,
    experiment_name: str,
    extra: str,
) -> None:
    """Saves the state dict of model in models_path/model_name/experiment_name/extra.pth.

    Args:
        model (torch.nn.Module): Model to save.
        models_path (PosixPath): Path the models are saved to.
        models_name (str): Name of the model to save.
        experiment_name (str): Name of the experiment to save (used as a subfolder).
        extra (str): Extra information concerning the training (used as name of file saved).
    """
    models_dir = models_path / model_name / experiment_name
    models_dir.mkdir(exist_ok=True, parents=True)

    MODEL_NAME = f"{extra}.pth"
    model_save_path = models_dir / MODEL_NAME

    print(f"\nSaving model to: {model_save_path}\n")

    torch.save(
        obj=model.state_dict(),
        f=model_save_path
    )


def save_results(
    root_dir: Path,
    models_name: str,
    experiment_name: str,
    extra: str,
    metrics: Dict[str, float],
) -> None:
    """Saves the metrics of the experiment in root_dir/debug/metrics/models_name/experiment_name.

    Args:
        root_dir (Path): The root directory of the project.
        models_name (str): The name of the model.
        experiment_name (str): The name of the experiment.
        extra (str): Extra information concerning the training (used as name of file saved).
        metrics (Dict[str, float]): The metrics of the experiment (classification metrics, loss,
            duration).
    """
    metrics_dir = root_dir / "debug" / "metrics" / models_name / experiment_name
    metrics_dir.mkdir(exist_ok=True, parents=True)

    metrics_file = metrics_dir / f"{extra}.txt"

    print(f"[INFO] Saving metrics to: {metrics_file}")

    with open(metrics_file, "w") as f:
        for key, metric in metrics.items():
            if key == "time":
                f.write(f"{key}: {get_time(metric)}\n")
            elif key == "loss":
                f.write(f"{key}: {metric:.4f}\n")
            else:
                f.write(f"{key}: {metric * 100:.2f}%\n")
