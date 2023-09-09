from typing import Dict, Iterator, Union

import torch.optim
from torch import optim
from typing_extensions import Self


class GetOptimizer:
    """Class that returns an optimizer based on the optimizer_name parameter. The list of available
    optimizers is: ["adam", "sgd"].

    In addition, the class also returns a scheduler based on the scheduler_name parameter. The list
    of available schedulers is: ["steplr"].

    Attributes:
        optimizer_name (str): String that identifies the optimizer to be used.
        params (Iterator[torch.nn.Parameter]): List of parameters of the model to be optimized.
        config (Union[Dict[str, float], None]): Dictionary with the configuration of the optimizer. 
        scheduler_name (Union[str, None]): String that identifies the scheduler to be used.
        scheduler_config (Union[Dict, None]): Dictionary with the configuration of the scheduler.

    Methods:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        _sgd: Returns a SGD optimizer.
        _adam: Returns an Adam optimizer.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Schedulers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        _steplr: Returns a StepLR scheduler.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get Transforms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        get_optimizer: Returns the optimizer based on the optimizer_name parameter.
        get_scheduler: Returns the scheduler based on the scheduler_name parameter.
    """

    def __init__(
        self: Self,
        params: Iterator[torch.nn.Parameter],
        optimizer_name: str,
        config: Union[Dict[str, float], None] = None,
        scheduler_name: Union[str, None] = None,
        scheduler_config: Union[Dict, None] = None,
    ) -> None:
        """Initializes the GetOptimizer class.

        Args:
            self (Self): GetOptimizer instance.
            params (List[torch.nn.Parameter]): List of parameters of the model to be optimized 
            optimizer_name (str): String that identifies the optimizer to be used (should match
                with a method implemented below, e.g. "sgd", "adam").
            config (Union[Dict[str, float], None], optional): Dictionary with the configuration 
                of the optimizer. Defaults to None.
            scheduler_name (Union[str, None], optional): String that identifies the scheduler to
                be used (should match with a method implemented below, e.g. "steplr"). Defaults to
                None.
            scheduler_config (Union[Dict, None], optional): Dictionary with the configuration of
                the scheduler. Defaults to None.
        """
        self.optimizer_name = optimizer_name
        self.params = params
        self.config = config
        self.scheduler_name = scheduler_name
        self.scheduler_config = scheduler_config

    def _sgd_optimizer(
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
        self.config.setdefault("momentum", 0.9)

        print(
            f"[INFO] Using SGD optimizer with lr={self.config['learning_rate']}, "
            f"momentum={self.config['momentum']} and "
            f"weight_decay={self.config['weight_decay']}."
        )

        return optim.SGD(
            params=self.params,
            lr=self.config["learning_rate"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"],
        )

    def _adam_optimizer(
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
        optimizer_method_name = f"_{self.optimizer_name}_optimizer"
        optimizer_method = getattr(self, optimizer_method_name, None)

        if optimizer_method is not None and callable(optimizer_method):
            self.optimizer = optimizer_method()
            return self.optimizer
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' is not supported.")

    def _steplr_scheduler(
        self: Self,

    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Returns a StepLR scheduler.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: StepLR scheduler.
        """
        if self.scheduler_config is None:
            self.scheduler_config = {}
        self.scheduler_config.setdefault("step_size", 5)
        self.scheduler_config.setdefault("gamma", 0.5)
        self.scheduler_config.setdefault("verbose", False)

        print(
            "[INFO] Using StepLR scheduler with "
            f"step_size={self.scheduler_config['step_size']}, "
            f"gamma={self.scheduler_config['gamma']} and "
            f"verbose={self.scheduler_config['verbose']}."
        )

        return optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=self.scheduler_config["step_size"],
            gamma=self.scheduler_config["gamma"],
            verbose=self.scheduler_config["verbose"],
        )

    def get_scheduler(
        self: Self,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Returns the scheduler based on the scheduler_name attribute.

        Raises:
            ValueError: When the scheduler_name does not correspond to any scheduler method.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: Scheduler instance.
        """
        scheduler_method_name = f"_{self.scheduler_name}_scheduler"
        scheduler_method = getattr(self, scheduler_method_name, None)

        if scheduler_method is not None and callable(scheduler_method):
            return scheduler_method()
        else:
            raise ValueError(f"Scheduler '{self.scheduler_name}' is not supported.")
