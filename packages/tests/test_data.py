
from packages.utils.configuration import data_dir
from packages.utils.load_data import plot_transformed_images
from packages.utils.transforms import GetTransforms


def test_transform(
    transform_name: str,
    **kwargs,
) -> None:
    """Test the train transform on the data, since test transform is a simple resize of
    the image (same as the one in train transform).

    Args:
        transform_name (str): The name of the transform to be tested.
        kwargs (dict): The configuration of the transform.
    """
    transform_config = kwargs.pop("transform_config", None)

    image_path_list = list(data_dir.glob("*/*/*.jpg"))

    # Create image transform
    transform_obj = GetTransforms(
        transform_name=transform_name,
        config=transform_config,
    )

    # Visualize  the results
    plot_transformed_images(
        image_paths=image_path_list,
        transform=transform_obj.get_train_transform(),
        transform_name=transform_obj.transform_name,
        n=10
    )
