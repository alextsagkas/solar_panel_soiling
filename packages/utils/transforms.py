from typing import Dict, Union

from torchvision import transforms
from typing_extensions import Self


class GetTransforms:
    """Get the transforms for the data. The experimentation is done on the train data,
    since the test data only resize resizing of the image and turning it to tensor.

    Args:
        transform_name (str, optional): Name of the transform to use. Defaults to "simple".
        config (Union[Dict[str, float], None], optional): Dictionary with the configuration
            of the transform. Defaults to None.

    Attributes:
        transform_name (str): Name of the transform to use.
        config (Union[Dict[str, float], None]): Dictionary with the configuration of the transform.

    Methods:
        _simple: Resize and randomly flip the image horizontally, then convert it to a tensor with
            values between 0 and 1.
        _trivial: Resize and apply trivial augment wide transform (https://arxiv.org/abs/2103.10158)
            Then convert it to a tensor with values between 0 and 1.
        get_train_transform: Returns the train transform based on the transform_name attribute.
        get_test_transform: Returns the test transform (default to the simple transform).
    """

    def __init__(
        self: Self,
        transform_name: str = "simple",
        config: Union[Dict[str, float], None] = None,
    ) -> None:
        """Initialize the transform object.

        Args:
            transform_name (str, optional): Name that defines which transform to use on the
                training data. Defaults to "simple".
            config (Union[Dict[str, float], None], optional): Dictionary with the configuration
                of the transform. Defaults to None.
        """

        self.transform_name = transform_name
        self.config = config

    def _simple_test(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and converts it to a tensor with values between 0 and 1.

        Returns:
            transforms.transforms.Compose: Simple transform.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("resize_size", 64)

        print(f"simple transform with resize_size={self.config['resize_size']}.")

        return transforms.Compose([
            transforms.Resize(
                size=(
                    self.config["resize_size"],
                    self.config["resize_size"]
                )
            ),
            transforms.ToTensor()
        ])

    def _simple_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and randomly flip the image horizontally, then convert it to a tensor with 
        values between 0 and 1.

        Returns:
            transforms.transforms.Compose: Simple transform.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("resize_size", 64)
        self.config.setdefault("random_horizontal_flip", 0.5)

        print(
            f"simple transform with resize_size={self.config['resize_size']} and "
            f"random_horizontal_flip={self.config['random_horizontal_flip']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=(
                    self.config["resize_size"],
                    self.config["resize_size"]
                )
            ),
            transforms.RandomHorizontalFlip(
                p=self.config["random_horizontal_flip"]
            ),
            transforms.ToTensor()
        ])

    def _trivial_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and apply trivial augment wide transform (https://arxiv.org/abs/2103.10158).
            Then convert it to a tensor with values between 0 and 1.

        Returns:
            transforms.transforms.Compose: Trivial transform.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("resize_size", 64)
        self.config.setdefault("num_magnitude_bins", 31)
        self.config.setdefault("random_rotation", 0)

        print(
            f"trivial transform with "
            f"resize_size={self.config['resize_size']}, "
            f"random rotation of {self.config['random_rotation']}˚ and "
            f"num_magnitude_bins={self.config['num_magnitude_bins']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=(
                    self.config["resize_size"],
                    self.config["resize_size"]
                )
            ),
            transforms.RandomRotation(
                degrees=self.config["random_rotation"]
            ),
            # Tuning-free Yet State-of-the-Art Data Augmentation
            transforms.TrivialAugmentWide(
                num_magnitude_bins=int(self.config["num_magnitude_bins"]),
            ),
            transforms.ToTensor()
        ])

    def get_train_transform(
        self
    ) -> transforms.transforms.Compose:
        """Returns the train transform based on the transform_name attribute.

        Raises:
            ValueError: When the transform_name does not correspond to any transform method.

        Returns:
            transforms.transforms.Compose: Train transform.
        """
        transform_method_name = f"_{self.transform_name}_train"
        transform_method = getattr(self, transform_method_name, None)

        if transform_method is not None and callable(transform_method):
            print("[INFO] Train data augmentation: ", end="")
            return transform_method()
        else:
            raise ValueError(f"Transform name {self.transform_name} is not supported.")

    def get_test_transform(
        self
    ) -> transforms.transforms.Compose:
        """Returns the test transform.

        Returns:
            transforms.transforms.Compose: Test simple transform (do not play with test data).
        """
        print("[INFO] Test data augmentation: ", end="")
        return self._simple_test()
