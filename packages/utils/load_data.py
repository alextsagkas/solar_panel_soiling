import os
import random
import torch
from pathlib import Path
from PIL import Image
from typing import Union
from pathlib import PosixPath
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.utils.data


def plot_transformed_images(
        image_paths: list[Path],
        transform: transforms.transforms.Compose,
        n: int = 3,
        seed: Union[int, None] = None) -> None:
    """Saves a series of random images from image_paths to the debug/data_transforms folder.

    Will open n image paths from image_paths, transform them
    with transform and save them one by one.

    Args:
        image_paths (list[str] | list[pathlib.PosixPath]): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to None.
    """
    # Set seed
    if seed != None:
        random.seed(seed)

    # Get random images
    random_image_paths = random.sample(image_paths, k=n)

    # Create folder to store results
    debug_folder = image_paths[0].parents[3] / "debug" / "data_transforms"
    os.makedirs(debug_folder, exist_ok=True)

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)

            # Original Image
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            transformed_image = torch.tensor(transform(f)).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            fig.savefig(str(debug_folder) + f"/{image_path.stem}_transformed.jpg")


def get_dataloader(
    dir: str,
    data_transform: transforms.transforms.Compose,
    BATCH_SIZE: int,
    NUM_WORKERS: int = 1,
    shuffle: bool = False,
) -> tuple[torch.utils.data.DataLoader, list[str]]:
    """Creates a dataset and passes it ot a dataloader, which is return in addition to class names.

    Args:
        dir (str): Directory of the data.
        data_transform (transforms.transforms.Compose): Transforms to apply to the data.
        BATCH_SIZE (int): Batch size.
        NUM_WORKERS (int, optional): The number workers that load the data (usually equals 
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

    dataloader = DataLoader(
        dataset=data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=shuffle
    )

    return dataloader, class_names


def get_dataloaders(
    train_dir: str,
    train_transform: transforms.transforms.Compose,
    test_dir: str,
    test_transform: transforms.transforms.Compose,
    BATCH_SIZE: int,
    NUM_WORKERS: int = 1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns iterables (Dataloaders) on train and test data.

    Args:
        train_dir (str): Training data directory.
        train_transform (transforms.transforms.Compose): Train data transforms.
        test_dir (str): Test data directory.
        test_transform (transforms.transforms.Compose): Test data transforms.
        BATCH_SIZE (int): Batch size.
        NUM_WORKERS (int, optional): Workers used to load the data (usually same 
            as the number of CPU cores). Defaults to 1.

    Returns:
      tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: List of train and test dataloader
    """
    train_dataloader, _ = get_dataloader(
        dir=train_dir,
        data_transform=train_transform,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=NUM_WORKERS,
        shuffle=True
    )

    test_dataloader, _ = get_dataloader(
        dir=test_dir,
        data_transform=test_transform,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=NUM_WORKERS,
        shuffle=False
    )

    return train_dataloader, test_dataloader
