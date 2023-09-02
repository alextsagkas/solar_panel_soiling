import os
from typing import List

import torch
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter

from packages.utils.configuration import tensorboard_dir


def create_writer(
    timestamp_list: List[str],
) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to the directory:
    debug/runs/YYYY-MM-DD/HH-MM-SS/"

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

    # Create tensorboard directory if it doesn't exist
    tensorboard_dir.mkdir(exist_ok=True, parents=True)

    log_dir = os.path.join(tensorboard_dir, *timestamp_list)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")

    return SummaryWriter(log_dir=log_dir)
