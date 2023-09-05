
from typing import List

from packages.utils.configuration import data_dir
from packages.utils.load_data import plot_transformed_images
from packages.utils.transforms import GetTransforms


def test_transform(
    timestamp_list: List[str],
    **kwargs,
) -> None:
    """Test the train and test transform on the data. The results are saved in debug/
    data_transforms/transform_name. 

    The hyperparameters provided in the function should contain only train or test transform 
    infos, and not both. An example is provided below:
        hyperparameters = {
            "timestamp_list": timestamp_list,
            "test_name": test_name,
            "train_transform_name": "resnet18",
            "train_transform_config": {"random_rotation": 180},
        }

    Args:
        timestamp_list (List[str]): The timestamp of the test. Used to create subfolders.
        kwargs (dict): The configuration of the test.
    """
    train_transform_name = kwargs.pop("train_transform_name", None)
    train_transform_config = kwargs.pop("train_transform_config", None)
    test_transform_name = kwargs.pop("test_transform_name", None)
    test_transform_config = kwargs.pop("test_transform_config", None)

    # Get all the image names inside the data/ folder
    image_path_list = list(data_dir.glob("*/*/*.jpg"))

    # Create image transform
    transform_obj = GetTransforms(
        train_transform_name=train_transform_name,
        train_config=train_transform_config,
        test_transform_name=test_transform_name,
        test_config=test_transform_config,
    )

    transform = (
        transform_obj.get_train_transform() if train_transform_name else
        transform_obj.get_test_transform()
    )
    transform_name = train_transform_name if train_transform_name else test_transform_name
    # Visualize  the results
    plot_transformed_images(
        image_paths=image_path_list,
        transform=transform,
        transform_name=transform_name,
        timestamp_list=timestamp_list,
        n=10,
    )
