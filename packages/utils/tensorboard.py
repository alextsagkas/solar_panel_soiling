import os
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter


def create_writer(
    experiment_name: str,
    model_name: str,
    transform_name: str,
    extra: Union[str, None] = None
) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/transform_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        transform_name (str): Name of transform.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """

    path = Path("/Users/alextsagkas/Document/Office/solar_panels/debug/runs")
    os.makedirs(path, exist_ok=True)

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join(path, timestamp, model_name, experiment_name, transform_name, extra)
    else:
        log_dir = os.path.join(path, timestamp, model_name, experiment_name, transform_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")

    return SummaryWriter(log_dir=log_dir)
