from torchvision import transforms


class GetTransforms:
    """Get the transforms for the data. The experimentation is done on the train data,
    since the test data only resize resizing of the image.

    Args:
        transform_name (str, optional): Name of the transform to use. Defaults to "simple".

    Attributes:
        transform_name_list (list[str]): List of transform names.
        transform_name (str): Name of the transform to use.
        simple_train_transform (transforms.transforms.Compose): Resize and randomly flip 
            the image horizontally, then convert it to a tensor with values between 0 and 1.
        trivial_train_transform (transforms.transforms.Compose): Resize and apply trivial augment
            wide transform (https://arxiv.org/abs/2103.10158) to the image, then convert it to a
            tensor with values between 0 and 1.
        simple_test_transform (transforms.transforms.Compose): Resize and convert the image 
            to a tensor with values between 0 and 1.

    Methods:
        get_train_transform: Returns the train transform.
        get_test_transform: Returns the test transform.
    """

    def __init__(
        self,
        transform_name: str = "simple",
    ) -> None:
        """Initialize the transform object.

        Args:
            transform_name (str, optional): Name that defines which transform to use on the
            training data. Defaults to "simple".

        Raises:
            ValueError: If the transform name is not in the list of transform names.
        """
        self.transform_name_list = ["simple", "trivial"]

        if transform_name not in self.transform_name_list:
            raise ValueError(f"transform_name must be one of {self.transform_name_list}")
        else:
            self.transform_name = transform_name
            print(
                f"[INFO] Using '{self.transform_name}' transform for training data and 'simple' transform for test data.")

        # Different Train Transforms
        self.simple_train_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        self.trivial_train_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            # Tuning-free Yet State-of-the-Art Data Augmentation
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])

        # Test Transform
        self.simple_test_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])

    def get_train_transform(self) -> transforms.transforms.Compose:  # type: ignore
        """Returns the train transform.

        Returns:
            transforms.transforms.Compose: Train transform based on 'transform_name' value.
        """
        transform_dict = {
            "simple": self.simple_train_transform,
            "trivial": self.trivial_train_transform,
        }

        if self.transform_name in transform_dict:
            return transform_dict[self.transform_name]

    def get_test_transform(self) -> transforms.transforms.Compose:
        """Returns the test transform.

        Returns:
            transforms.transforms.Compose: Test simple transform (do not play with test data).
        """
        return self.simple_test_transform
