import os
import random
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from packages.utils.configuration import data_transforms_dir


def plot_transformed_images(
        image_paths: list[Path],
        transform: transforms.transforms.Compose,
        transform_name: str,
        timestamp_list: List[str],
        n: int = 3,
        seed: Union[int, None] = None
) -> None:
    """Saves a series of random images from image_paths to the debug/data_transforms/transform_name/
    YYYY-MM-DD/HH-MM-SS/ folder.

    Will open n image paths from image_paths, transform them
    with transform and save them one by one.

    Args:
        image_paths (list[Path]): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        transform_name (str): Name of the transform to use as a subfolder for 
            saving the transformed images.
        timestamp_list (List[str]): Timestamp of the test.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to None.k
    """
    # Set seed
    if seed != None:
        random.seed(seed)

    # Get random images
    random_image_paths = random.sample(image_paths, k=n)

    # Create folder to store results
    debug_folder = data_transforms_dir / transform_name / timestamp_list[0] / timestamp_list[1]
    os.makedirs(debug_folder, exist_ok=True)

    print(f"[INFO] Saving transformed images to {debug_folder} folder")

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)

            # Original Image
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            transformed_image = transform(f).permute(1, 2, 0)  # type: ignore
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            fig.savefig(str(debug_folder) + f"/{image_path.stem}_transformed.jpeg")


def get_dataloader(
    dir: str,
    data_transform: transforms.transforms.Compose,
    batch_size: int,
    num_workers: int = 1,
    shuffle: bool = False,
) -> tuple[torch.utils.data.DataLoader, list[str]]:
    """Creates a dataset and passes it ot a dataloader, which is return in addition to class names.

    Args:
        dir (str): Directory of the data.
        data_transform (transforms.transforms.Compose): Transforms to apply to the data.
        batch_size (int): Batch size.
        num_workers (int, optional): The number workers that load the data (usually equals 
            to the cpu cores). Defaults to 1.
        shuffle (bool, optional): Shuffle the data or not (usually shuffle only the training
            data). Defaults to False.

    Returns:
        tuple[
            torch.utils.data.DataLoader,
            list[str]
        ]: Dataloader of the data and the class names.
    """
    data = datasets.ImageFolder(
        root=dir,
        transform=data_transform,
        target_transform=None
    )

    class_names = data.classes

    print(f"[INFO] Using {num_workers} workers to load data.")

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return dataloader, class_names
