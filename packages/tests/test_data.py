from pathlib import Path

from packages.utils.load_data import plot_transformed_images
from packages.utils.transforms import GetTransforms


def test_transform(
    data_dir: Path,
    transform_obj: GetTransforms
) -> None:
    """Test the train transform on the data, since test transform is a simple resize of
    the image (same as the one in train transform).

    Args:
        data_dir (Path): Directory where the data (images) are stored.
        transform_obj (GetTransforms): Transform object.
    """
    image_path_list = list(data_dir.glob("*/*/*.jpg"))

    # Create image transform
    data_transform = transform_obj.get_train_transform()

    # Visualize  the results
    plot_transformed_images(
        image_paths=image_path_list,
        transform=data_transform,
        transform_name=transform_obj.transform_name,
        n=10
    )
