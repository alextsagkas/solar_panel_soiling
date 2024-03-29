from typing import Dict, Union

from torchvision import transforms
from typing_extensions import Self


class GetTransforms:
    """Get the transforms for training and testing data using the corresponding names for them.  The transforms can also be optionally configured through the train_config and test_config attributes.

    **Attributes:**
    
        train_transform_name : str
			Name of the transform to use on the train data.
        train_config : Union[Dict[str, float], None]
			Dictionary with the configuration of the train transform.
        test_transform_name : str
			Name of the transform to use on the test data.
        test_config : Union[Dict[str, float], None]
			Dictionary with the configuration of the test transform.

    **Methods:**

    -----------------------------------------------------------------------------------------------

        simple_train:
			Resize and randomly flip the image horizontally, then convert it to a tensor with values between 0 and 1.
        trivial_train:
			Resize and apply trivial augment wide transform (https://arxiv.org/abs/2103.10158). Then convert it to a tensor with values between 0 and 1.
        resnet_train:
			Resize and crops the image before applying data augmentation (horizontal flip & rotation). In the end the image is converted to tensor with values between 0 and 1 and the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.
        efficientnet_train:
			Resize and central crops the image before applying data augmentation (horizontal flip & rotation). In the end the image is converted to tensor with values between 0 and 1 and the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.
        mobilenet_train:
			Resize and central crops the image before applying data augmentation (horizontal flip & rotation). In the end the image is converted to tensor with values between 0 and 1 and the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.

    -----------------------------------------------------------------------------------------------

        simple_test:
			Resize and converts it to a tensor with values between 0 and 1.
        resnet_test:
			Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.
        efficientnet_test:
			Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.
        mobilenet_test:
			Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.

    -----------------------------------------------------------------------------------------------

        get_train_transform: 
            Returns the train transform based on the train_transform_name  attribute.
        get_test_transform: 
            Returns the test transform based on the test_transform_name attribute.
    """

    def __init__(
        self: Self,
        train_transform_name: str = "simple",
        train_config: Union[Dict, None] = None,
        test_transform_name: str = "simple",
        test_config: Union[Dict, None] = None,
    ) -> None:
        """Initialize the transform object.

        **Args**:

            train_transform_name : str, optional
                Name of the transform to use on the train data. Defaults to "simple".
            train_config : Union[Dict, None], optional
                Dictionary with the configuration of the train transform. Defaults to None.
            test_transform_name : str, optional
                Name of the transform to use on the test data. Defaults to "simple".
            test_config : Union[Dict, None], optional
                Dictionary with the configuration of the test transform. Defaults to None.
        """
        self.train_transform_name = train_transform_name
        self.train_config = train_config
        self.test_transform_name = test_transform_name
        self.test_config = test_config

    def _simple_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and randomly flip the image horizontally, then convert it to a tensor with values between 0 and 1.

        **Returns**:

            transforms.transforms.Compose: 
                Simple transform.
        """
        if self.train_config is None:
            self.train_config = {}
        self.train_config.setdefault("resize_size", 64)
        self.train_config.setdefault("random_horizontal_flip", 0.5)

        print(
            f"simple transform with resize_size={self.train_config['resize_size']} and "
            f"random_horizontal_flip={self.train_config['random_horizontal_flip']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=self.train_config["resize_size"]
            ),
            transforms.RandomHorizontalFlip(
                p=self.train_config["random_horizontal_flip"]
            ),
            transforms.ToTensor()
        ])

    def _trivial_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and apply trivial augment wide transform (https://arxiv.org/abs/2103.10158). Then convert it to a tensor with values between 0 and 1.

        **Returns**:

            transforms.transforms.Compose: Trivial transform.
        """
        if self.train_config is None:
            self.train_config = {}
        self.train_config.setdefault("resize_size", 64)
        self.train_config.setdefault("num_magnitude_bins", 31)
        self.train_config.setdefault("random_rotation", 0)

        print(
            f"trivial transform with "
            f"resize_size={self.train_config['resize_size']}, "
            f"random rotation of {self.train_config['random_rotation']}˚ and "
            f"num_magnitude_bins={self.train_config['num_magnitude_bins']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=self.train_config["resize_size"]
            ),
            transforms.RandomRotation(
                degrees=self.train_config["random_rotation"]
            ),
            transforms.TrivialAugmentWide(
                num_magnitude_bins=int(self.train_config["num_magnitude_bins"]),
            ),
            transforms.ToTensor()
        ])

    def _resnet_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and randomly crops the image before applying data augmentation (horizontal flip,  trivial transform (optional - avoid if num_magnitude_bins=0) & rotation). In the end the  image is converted to tensor with values between 0 and 1 and the mean = [0.485, 0.456, 0. 406] and std = [0.229, 0.224, 0.225] are subtracted.

        **Args**:

            self : Self
			GetTransforms instance.

        **Returns**:

            transforms.transforms.Compose: 
                The resnet train transform.
        """
        if self.train_config is None:
            self.train_config = {}
        self.train_config.setdefault("resize_size", 256)
        self.train_config.setdefault("crop_size", 224)
        self.train_config.setdefault("num_magnitude_bins", 0)
        self.train_config.setdefault("random_horizontal_flip", 0.5)
        self.train_config.setdefault("random_rotation", 0)

        print(
            "resnet transform with "
            f"resize_size={self.train_config['resize_size']}, "
            f"crop_size={self.train_config['crop_size']}, "
            f"num_magnitude_bins={self.train_config['num_magnitude_bins']}, "
            f"random_horizontal_flip={self.train_config['random_horizontal_flip']} and "
            f"random_rotation={self.train_config['random_rotation']}."
        )

        # Create transform function
        transforms_list = [
            transforms.Resize(
                size=self.train_config["resize_size"]
            ),
            transforms.RandomResizedCrop(
                size=self.train_config["crop_size"]
            ),
            transforms.RandomHorizontalFlip(
                p=self.train_config["random_horizontal_flip"]
            ),
            transforms.RandomRotation(
                degrees=self.train_config["random_rotation"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

        # Add trivial augment wide transform if num_magnitude_bins > 0
        if self.train_config["num_magnitude_bins"] > 0:
            transforms_list.insert(
                2,
                transforms.TrivialAugmentWide(
                    num_magnitude_bins=self.train_config["num_magnitude_bins"],
                )
            )

        return transforms.Compose(transforms_list)

    def _efficientnet_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and central crops the image before applying data augmentation (horizontal flip, & random rotation). In the end the image is converted to tensor with values between 0 and 1  and the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted. The  values of mean and std can be optionally configured through the train_config attribute.

        **Args**:

            self : Self
			GetTransforms instance.

        **Returns**:

            transforms.transforms.Compose: The efficientnet train transform.
        """
        if self.train_config is None:
            self.train_config = {}
        self.train_config.setdefault("resize_size", 256)
        self.train_config.setdefault("crop_size", 224)
        self.train_config.setdefault("random_horizontal_flip", 0.5)
        self.train_config.setdefault("random_rotation", 0)
        self.train_config.setdefault("mean", [0.485, 0.456, 0.406])
        self.train_config.setdefault("std", [0.229, 0.224, 0.225])

        print(
            "efficientnet transform with "
            f"resize_size={self.train_config['resize_size']}, "
            f"crop_size={self.train_config['crop_size']}, "
            f"random_horizontal_flip={self.train_config['random_horizontal_flip']}, "
            f"random_rotation={self.train_config['random_rotation']}, "
            f"mean={self.train_config['mean']} and "
            f"std={self.train_config['std']}."
        )

        # Create transform function
        transforms_list = [
            transforms.Resize(
                size=self.train_config["resize_size"],
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(
                size=self.train_config["crop_size"]
            ),
            transforms.RandomHorizontalFlip(
                p=self.train_config["random_horizontal_flip"]
            ),
            transforms.RandomRotation(
                degrees=self.train_config["random_rotation"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.train_config["mean"],
                std=self.train_config["std"],
            )
        ]

        return transforms.Compose(transforms_list)

    def _mobilenet_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and central crops the image before applying data augmentation (horizontal flip, & random rotation). In the end the image is converted to tensor with values between 0 and 1 and the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted. The values of mean and std can be optionally configured through the train_config attribute.

        **Args**:

            self : Self
			GetTransforms instance.

        **Returns**:

            transforms.transforms.Compose: 
                The mobilenet train transform.
        """
        if self.train_config is None:
            self.train_config = {}
        self.train_config.setdefault("resize_size", 232)
        self.train_config.setdefault("crop_size", 224)
        self.train_config.setdefault("random_horizontal_flip", 0.5)
        self.train_config.setdefault("random_rotation", 0)
        self.train_config.setdefault("mean", [0.485, 0.456, 0.406])
        self.train_config.setdefault("std", [0.229, 0.224, 0.225])

        print(
            "efficientnet transform with "
            f"resize_size={self.train_config['resize_size']}, "
            f"crop_size={self.train_config['crop_size']}, "
            f"random_horizontal_flip={self.train_config['random_horizontal_flip']}, "
            f"random_rotation={self.train_config['random_rotation']}, "
            f"mean={self.train_config['mean']} and "
            f"std={self.train_config['std']}."
        )

        # Create transform function
        transforms_list = [
            transforms.Resize(
                size=self.train_config["resize_size"],
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(
                size=self.train_config["crop_size"]
            ),
            transforms.RandomHorizontalFlip(
                p=self.train_config["random_horizontal_flip"]
            ),
            transforms.RandomRotation(
                degrees=self.train_config["random_rotation"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.train_config["mean"],
                std=self.train_config["std"],
            )
        ]

        return transforms.Compose(transforms_list)

    def _shufflenet_train(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and central crops the image before applying data augmentation (horizontal flip).  In the end the image is converted to tensor with values between 0 and 1 and the mean = [0. 485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted. The values of mean and std can be optionally configured through the train_config attribute.

        **Args**:

            self : Self
			GetTransforms instance.

        **Returns**:

            transforms.transforms.Compose:
                The shufflenet train transform.
        """
        if self.train_config is None:
            self.train_config = {}
        self.train_config.setdefault("resize_size", 256)
        self.train_config.setdefault("crop_size", 224)
        self.train_config.setdefault("random_horizontal_flip", 0.5)
        self.train_config.setdefault("mean", [0.485, 0.456, 0.406])
        self.train_config.setdefault("std", [0.229, 0.224, 0.225])

        print(
            "shufflenet transform with "
            f"resize_size={self.train_config['resize_size']}, "
            f"crop_size={self.train_config['crop_size']}, "
            f"random_horizontal_flip={self.train_config['random_horizontal_flip']}, "
            f"mean={self.train_config['mean']} and "
            f"std={self.train_config['std']}."
        )

        # Create transform function
        transforms_list = [
            transforms.Resize(
                size=self.train_config["resize_size"],
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(
                size=self.train_config["crop_size"]
            ),
            transforms.RandomHorizontalFlip(
                p=self.train_config["random_horizontal_flip"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.train_config["mean"],
                std=self.train_config["std"],
            )
        ]

        return transforms.Compose(transforms_list)

    def get_train_transform(
        self
    ) -> transforms.transforms.Compose:
        """Returns the train transform based on the train_transform_name attribute.

        **Raises:**

            ValueError: 
                When the transform_name does not correspond to any transform method.

        **Returns**:

            transforms.transforms.Compose: 
                Train transform.
        """
        transform_method_name = f"_{self.train_transform_name}_train"
        transform_method = getattr(self, transform_method_name, None)

        if transform_method is not None and callable(transform_method):
            print("[INFO] Train data augmentation: ", end="")
            return transform_method()
        else:
            raise ValueError(f"Transform name {self.train_transform_name} is not supported.")

    def _simple_test(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and converts it to a tensor with values between 0 and 1.

        **Returns**:

            transforms.transforms.Compose: Simple transform.
        """
        if self.test_config is None:
            self.test_config = {}
        self.test_config.setdefault("resize_size", 64)

        print(f"simple transform with resize_size={self.test_config['resize_size']}.")

        return transforms.Compose([
            transforms.Resize(
                size=self.test_config["resize_size"],
            ),
            transforms.ToTensor()
        ])

    def _resnet_test(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted.

        **Args**:

            self : Self
			The object itself.

        **Returns**:

            transforms.transforms.Compose:
                The resnet test transform.
        """
        if self.test_config is None:
            self.test_config = {}
        self.test_config.setdefault("resize_size", 256)
        self.test_config.setdefault("crop_size", 224)

        print(
            f"{self.test_transform_name} transform with "
            f"resize_size={self.test_config['resize_size']}, "
            f"crop_size={self.test_config['crop_size']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=self.test_config["resize_size"],
            ),
            transforms.CenterCrop(
                size=self.test_config["crop_size"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _efficientnet_test(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted. The values of mean and std can be optionally configured through the test_config attribute.

        **Args**:

            self : Self
			The object itself.

        **Returns**:

            transforms.transforms.Compose: 
                The efficientnet test transform.
        """
        if self.test_config is None:
            self.test_config = {}
        self.test_config.setdefault("resize_size", 256)
        self.test_config.setdefault("crop_size", 224)
        self.test_config.setdefault("mean", [0.485, 0.456, 0.406])
        self.test_config.setdefault("std", [0.229, 0.224, 0.225])

        print(
            f"{self.test_transform_name} transform with "
            f"resize_size={self.test_config['resize_size']}, "
            f"crop_size={self.test_config['crop_size']}, "
            f"mean={self.test_config['mean']} and "
            f"std={self.test_config['std']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=self.test_config["resize_size"],
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(
                size=self.test_config["crop_size"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.test_config["mean"],
                std=self.test_config["std"],
            )
        ])

    def _mobilenet_test(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted. The values of mean and std can be optionally configured through the test_config attribute.

        **Args**:

            self : Self
			The object itself.

        **Returns**:

            transforms.transforms.Compose: 
                The mobilenet test transform.
        """
        if self.test_config is None:
            self.test_config = {}
        self.test_config.setdefault("resize_size", 232)
        self.test_config.setdefault("crop_size", 224)
        self.test_config.setdefault("mean", [0.485, 0.456, 0.406])
        self.test_config.setdefault("std", [0.229, 0.224, 0.225])

        print(
            f"{self.test_transform_name} transform with "
            f"resize_size={self.test_config['resize_size']}, "
            f"crop_size={self.test_config['crop_size']}, "
            f"mean={self.test_config['mean']} and "
            f"std={self.test_config['std']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=self.test_config["resize_size"],
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(
                size=self.test_config["crop_size"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.test_config["mean"],
                std=self.test_config["std"],
            )
        ])

    def _shufflenet_test(
        self: Self,
    ) -> transforms.transforms.Compose:
        """Resize and crops the image before converting it to a tensor with values between 0 and 1. In the end the mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] are subtracted. The values of mean and std can be optionally configured through the test_config attribute.

        **Args**:

            self : Self
			The object itself.

        **Returns**:

            transforms.transforms.Compose: 
                The shufflenet test transform.
        """
        if self.test_config is None:
            self.test_config = {}
        self.test_config.setdefault("resize_size", 256)
        self.test_config.setdefault("crop_size", 224)
        self.test_config.setdefault("mean", [0.485, 0.456, 0.406])
        self.test_config.setdefault("std", [0.229, 0.224, 0.225])

        print(
            f"{self.test_transform_name} transform with "
            f"resize_size={self.test_config['resize_size']}, "
            f"crop_size={self.test_config['crop_size']}, "
            f"mean={self.test_config['mean']} and "
            f"std={self.test_config['std']}."
        )

        return transforms.Compose([
            transforms.Resize(
                size=self.test_config["resize_size"],
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(
                size=self.test_config["crop_size"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.test_config["mean"],
                std=self.test_config["std"],
            )
        ])

    def get_test_transform(
        self
    ) -> transforms.transforms.Compose:
        """Returns the test transform based on the test_transform_name attribute.

        **Raises:**

            ValueError: 
                When the transform_name does not correspond to any transform method.

        **Returns**:

            transforms.transforms.Compose: 
                Train transform.
        """
        transform_method_name = f"_{self.test_transform_name}_test"
        transform_method = getattr(self, transform_method_name, None)

        if transform_method is not None and callable(transform_method):
            print("[INFO] Test data augmentation: ", end="")
            return transform_method()
        else:
            raise ValueError(f"Transform name {self.test_transform_name} is not supported.")
