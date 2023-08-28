import os
import random
from PIL import Image
from typing import Union
from pathlib import PosixPath
import matplotlib.pyplot as plt
from torchvision import transforms


def plot_transformed_images(
        image_paths: list[PosixPath],
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
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            fig.savefig(debug_folder / f"{image_path.stem}_transformed.jpg")
