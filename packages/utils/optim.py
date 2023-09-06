from typing import Dict, Iterator, Union

import torch.optim
from torch import optim
from typing_extensions import Self


class GetOptimizer:
    """Class that returns an optimizer based on the optimizer_name parameter. The list of available
    optimizers is: ["adam", "sgd"].

    Attributes:
        optimizer_name (str): String that identifies the optimizer to be used.
        params (Iterator[torch.nn.Parameter]): List of parameters of the model to be optimized.
        config (Union[Dict[str, float], None]): Dictionary with the configuration of the optimizer. 

    Methods:
        _sgd: Returns a SGD optimizer.
        _adam: Returns an Adam optimizer.
        get_optimizer: Returns the optimizer based on the optimizer_name parameter.
    """

    def __init__(
        self: Self,
        params: Iterator[torch.nn.Parameter],
        optimizer_name: str,
        config: Union[Dict[str, float], None] = None,
    ) -> None:
        """Initializes the GetOptimizer class.

        Args:
            self (Self): GetOptimizer instance.
            params (List[torch.nn.Parameter]): List of parameters of the model to be optimized 
            optimizer_name (str): String that identifies the optimizer to be used (should match
                with a method implemented below, e.g. "sgd", "adam").
            config (Union[Dict[str, float], None], optional): Dictionary with the configuration 
                of the optimizer. Defaults to None.
        """
        self.optimizer_name = optimizer_name
        self.params = params
        self.config = config

    def _sgd(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns a SGD optimizer.

        Returns:
            torch.optim.Optimizer: SGD optimizer.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("learning_rate", 1e-2)
        self.config.setdefault("weight_decay", 0.0)

        print(f"[INFO] Using SGD optimizer with lr={self.config['learning_rate']}")

        return optim.SGD(
            params=self.params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def _adam(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns an Adam optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("beta1", 0.9)
        self.config.setdefault("beta2", 0.999)
        self.config.setdefault("epsilon", 1e-8)
        self.config.setdefault("weight_decay", 0.0)

        print(
            f"[INFO] Using Adam optimizer with "
            f"lr={self.config['learning_rate']}, "
            f"beta1={self.config['beta1']}, "
            f"beta2={self.config['beta2']}, "
            f"epsilon={self.config['epsilon']} and "
            f"weight_decay={self.config['weight_decay']}."
        )

        return optim.Adam(
            params=self.params,
            lr=self.config["learning_rate"],
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=self.config["epsilon"],
            weight_decay=self.config["weight_decay"],
        )

    def get_optimizer(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns the optimizer based on the optimizer_name attribute.

        Returns:
            torch.optim.Optimizer: Optimizer instance.

        Raises:
            ValueError: When the optimizer_name does not correspond to any optimizer method.
        """
        optimizer_method_name = f"_{self.optimizer_name}"
        optimizer_method = getattr(self, optimizer_method_name, None)

        if optimizer_method is not None and callable(optimizer_method):
            return optimizer_method()
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' is not supported")
