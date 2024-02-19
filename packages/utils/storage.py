import ast
from pathlib import Path
from typing import Dict, List, Union

import torch

from packages.utils.configuration import config_dir, metrics_dir, models_dir
from packages.utils.time import get_time


def save_model(
    model: torch.nn.Module,
    timestamp_list: List[str],
    save_dir: Path = models_dir,
    extra: Union[str, None] = None,
) -> None:
    """Saves the state dict of model in save_dir/YYYY-MM-DD/HH-MM-SS_{extra}.pth.

    **Args:**

        model : torch.nn.Module
			Model to save.
        timestamp_list : List[str]
			List of timestamp (YYYY-MM-DD, HH-MM-SS).
        save_dir : Path, optional
			Directory where to save the model. Defaults to models_dir.
        extra : str
			Extra information concerning the training (used as name of file saved).
    """
    model_save_dir = save_dir / timestamp_list[0]
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

    **Args:**

        metrics : Dict[str, float]
			The metrics of the experiment (classification metrics, loss, duration).
        timestamp_list : List[str]
			List of timestamp (YYYY-MM-DD, HH-MM-SS).
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
            elif key == "epoch":
                f.write(f"{key}: {metric:.0f}\n")
            else:
                f.write(f"{key}: {metric * 100:.2f}%\n")


def save_hyperparameters(
    hyperparameters: Dict[str, Union[str, int, float]],
) -> None:
    """Saves the hyperparameters of the experiment in config_dir/YYYY-MM-DD/HH-MM-SS.txt.

    **Args:**

        hyperparameters : Dict[str, Union[str, int, float]]
			The hyperparameters of the experiment.
    """
    config_save_dir = config_dir / Path(hyperparameters["timestamp_list"][0])  # type: ignore
    config_save_dir.mkdir(exist_ok=True)

    config_path = config_save_dir / f"{hyperparameters['timestamp_list'][1]}.txt"  # type: ignore

    print(f"[INFO] Saving hyperparameters to: {config_path}.")

    # Write arguments on config.txt file
    with open(config_path, "w") as f:  # type: ignore
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")


def load_hyperparameters(
    test_timestamp_list: List[str],
) -> Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]:
    """Loads the hyperparameters of the experiment from config_dir/YYYY-MM-DD/HH-MM-SS.txt.

    **Args:**

        test_timestamp_list : List[str]
			List of timestamp (YYYY-MM-DD, HH-MM-SS).

    **Returns:**

        Dict[str, str]: hyperparameters used for the loaded experiment.
    """
    config_path = config_dir / test_timestamp_list[0] / f"{test_timestamp_list[1]}.txt"

    print(f"[INFO] Loading hyperparameters from: {config_path}.")

    # Create a dictionary with keys and values out of every line of config
    with open(config_path, "r") as file:
        lines = file.readlines()

    # Create the config dictionary
    config_dict = {}

    for line in lines:
        # Split each line into key and value using ':'
        key, value = line.strip().split(': ', 1)

        # Check if the value can be parsed as a dictionary
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If it's not a dictionary, use it as is
            pass

        # Add the key-value pair to the hyperparameters dictionary
        config_dict[key] = value

    return config_dict
