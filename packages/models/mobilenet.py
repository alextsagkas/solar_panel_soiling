import torch
import torchvision
from torch import nn
from typing_extensions import Self


class MobileNetV2(nn.Module):
    """Class that returns a MobileNetV2 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the MobileNetV2 model from IMAGENET1K_V2.
        model (torch.nn.Module): MobileNetV2 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V2 weights for MobileNetV2 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        # Load the model with the pretrained weights
        self.model = torchvision.models.mobilenet_v2(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(
                in_features=1280,
                out_features=2,
                bias=True
            )
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)


class MobileNetV3Small(nn.Module):
    """Class that returns a MobileNetV3Small model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the MobileNetV3Small model from IMAGENET1K_V1.
        model (torch.nn.Module): MobileNetV3Small model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for MobileNetV3Small and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.mobilenet_v3_small(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=576,
                out_features=512,  # 1024 or 512 (if reduced_tail = 0 or 1)
            ),
            nn.Hardswish(
                inplace=True
            ),
            nn.Dropout(
                p=0.2,
                inplace=True
            ),
            nn.Linear(
                in_features=512,
                out_features=2,
                bias=True
            )
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)


class MobileNetV3Large(nn.Module):
    """Class that returns a MobileNetV3Large model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the MobileNetV3Large model from IMAGENET1K_V2.
        model (torch.nn.Module): MobileNetV3Large model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V2 weights for MobileNetV3Large and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        # Load the model with the pretrained weights
        self.model = torchvision.models.mobilenet_v3_large(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        last_channel = 1280

        self.model.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=960,
                out_features=last_channel,
            ),
            nn.Hardswish(
                inplace=True
            ),
            nn.Dropout(
                p=0.2,
                inplace=True
            ),
            nn.Linear(
                in_features=last_channel,
                out_features=2,
                bias=True
            )
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)
