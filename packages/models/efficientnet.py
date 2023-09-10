import torch
import torchvision
from torch import nn
from typing_extensions import Self


class EfficientNetB0(nn.Module):
    """Class that returns a EfficientNet_B0 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the EfficientNet_B0 model from IMAGENET1K_V1.
        model (torch.nn.Module): EfficientNet_B0 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for EfficientNet_B0 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.efficientnet_b0(
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


class EfficientNetB1(nn.Module):
    """Class that returns a EfficientNet_B1 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the EfficientNet_B1 model from IMAGENET1K_V1.
        model (torch.nn.Module): EfficientNet_B1 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for EfficientNet_B1 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.efficientnet_b1(
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


class EfficientNetB2(nn.Module):
    """Class that returns a EfficientNet_B2 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the EfficientNet_B2 model from IMAGENET1K_V1.
        model (torch.nn.Module): EfficientNet_B2 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for EfficientNet_B2 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.efficientnet_b2(
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
                in_features=1408,
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


class EfficientNetB3(nn.Module):
    """Class that returns a EfficientNet_B3 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the EfficientNet_B3 model from IMAGENET1K_V1.
        model (torch.nn.Module): EfficientNet_B3 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for EfficientNet_B3 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.efficientnet_b3(
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
                in_features=1536,
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


class EfficientNetB6(nn.Module):
    """Class that returns a EfficientNet_B6 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the EfficientNet_B6 model from IMAGENET1K_V1.
        model (torch.nn.Module): EfficientNet_B6 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for EfficientNet_B6 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.efficientnet_b6(
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
                in_features=2304,
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
