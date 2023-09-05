import torch
from torch import nn
from typing_extensions import Self


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: https://poloclub.github.io/cnn-explainer/
    with dropout layers.
    """

    def __init__(
        self: Self,
        dropout_rate: float,
        input_shape: int,
        hidden_units: int,
        output_shape: int
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            self.dropout,
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            self.dropout,
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 16 * 16,
                out_features=output_shape
            )
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


class TinyVGGBatchnorm(nn.Module):
    """
    Model architecture copying TinyVGG from: https://poloclub.github.io/cnn-explainer/
    with dropout layers and batchnorm layers.
    """

    def __init__(
        self: Self,
        dropout_rate: float,
        input_shape: int,
        hidden_units: int,
        output_shape: int
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            self.dropout,
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            self.dropout,
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            self.dropout,
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            self.dropout,
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 16 * 16,
                out_features=output_shape
            )
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
