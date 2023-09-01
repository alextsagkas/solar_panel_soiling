import torch.optim
from torch import optim
from typing_extensions import Self


class GetOptimizer:
    """Class that returns an optimizer based on the optimizer_name parameter. The list of available
    optimizers is: ["adam", "sgd"].

    Args:
        model (torch.nn.Module): The model to be optimized.
        optimizer_name (str): String that identifies the optimizer to be used.
        learning_rate (float): Learning rate to be used by the optimizer.

    Attributes:
        optimizer_name_list (List[str]): List of available optimizers.
        optimizer_name (str): String that identifies the optimizer to be used.
        model_params (List[torch.nn.Parameter]): List of parameters of the model to be optimized.
        learning_rate (float): Learning rate to be used by the optimizer.

    Methods:
        get_optimizer: Returns the optimizer based on the optimizer_name parameter.
    """

    def __init__(
        self: Self,
        model: torch.nn.Module,
        optimizer_name: str,
        learning_rate: float,
    ) -> None:
        """Initializes the GetOptimizer class.

        Args:
            self (Self): GetOptimizer instance.
            model (torch.nn.Module): The model to be optimized.
            optimizer_name (str): String that identifies the optimizer to be used.
            learning_rate (float): Learning rate to be used by the optimizer.

        Raises:
            ValueError: When the optimizer_name is not one of the available optimizers.
            ValueError: When the learning_rate is not between 0 and 1.
        """
        self.optimizer_name_list = ["adam", "sgd"]
        if optimizer_name not in self.optimizer_name_list:
            raise ValueError(f"optimizer_name must be one of {self.optimizer_name_list}")
        else:
            self.optimizer_name = optimizer_name

        # list is used to be able to instantiate the optimizers in the get_optimizer method
        self.model_params = list(model.parameters())

        if learning_rate < 0 or learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        else:
            self.learning_rate = learning_rate

        print(
            f"[INFO] Using {self.optimizer_name} optimizer, "
            f"with {self.learning_rate} learning rate "
            f"on {model.model_name} model."
        )

    def get_optimizer(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns the optimizer based on the optimizer_name parameter.

        Args:
            self (Self): GetOptimizer instance.

        Returns:
            torch.optim.Optimizer: The optimizer to be used.
        """
        optimizer_dict = {
            "adam": optim.Adam(
                params=self.model_params,
                lr=self.learning_rate,
            ),
            "sgd": optim.SGD(
                params=self.model_params,
                lr=self.learning_rate,
            )
        }

        return optimizer_dict[self.optimizer_name]
