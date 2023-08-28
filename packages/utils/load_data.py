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


def get_dataloaders(
    train_dir: str,
    train_transform: transforms.transforms.Compose,
    test_dir: str,
    test_transform: transforms.transforms.Compose,
    BATCH_SIZE: int,
    NUM_WORKERS: int = 1,
):
    """Returns iterables (Dataloaders) on train and test data.

    Args:
        train_dir (Union[str, PosixPath]): training data directory
        train_transform (transforms.transforms.Compose): train data transforms
        test_dir (Union[str, PosixPath]): test data directory
        test_transform (transforms.transforms.Compose): test data transforms
        BATCH_SIZE (int): batch size
        NUM_WORKERS (int, optional): workers used to load the data (usually same 
            as the number of CPU cores). Defaults to 1.

    Returns:
        list[torch.utils.data.Dataloader]: list of train and test dataloader
    """
    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
        target_transform=None
    )
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True)

    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform,
        target_transform=None
    )
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)

    return train_dataloader, test_dataloader
