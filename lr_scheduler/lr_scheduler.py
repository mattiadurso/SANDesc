class LrManager:
    """
    Custom learning rate manager for various scheduling strategies.
    """

    def __init__(
        self,
        name: str = "constant",
        lr_max: float = 5e-3,
        lr_min: float = 1e-4,
        decay_steps: int = 100_000,
        max_iterations: int = 200_000,
        warmup_steps: int = 2048,
    ):
        self.name = name
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.decay_steps = decay_steps
        self.max_iterations = max_iterations
        self.lr_decay = (lr_min / lr_max) ** (1 / (decay_steps))
        self.warmup_steps = warmup_steps

    def get_lr(self, iteration: int = 0):
        if self.name == "constant":
            return self.lr_min

        elif self.name == "warmup_constant":
            return min(
                self.lr_max,
                self.lr_min
                + ((self.lr_max - self.lr_min) * (iteration / self.warmup_steps)),
            )

        elif self.name == "decay_constant":
            return max(self.lr_min, self.lr_max * (self.lr_decay**iteration))

        elif self.name == "warmup_decay_constant":
            self.decay = 0.99993
            if iteration < self.warmup_steps:
                return self.lr_min + (
                    (self.lr_max - self.lr_min) * (iteration / self.warmup_steps)
                )
            else:
                return max(
                    self.lr_min,
                    self.lr_max * (self.decay ** (iteration - self.warmup_steps)),
                )

        elif self.name == "linear_increase_linear_decrease_constant":
            if iteration < self.warmup_steps:
                return self.lr_min + (
                    (self.lr_max - self.lr_min) * (iteration / self.warmup_steps)
                )
            else:
                return max(
                    self.lr_min,
                    self.lr_max
                    - (
                        (self.lr_max - self.lr_min)
                        * ((iteration - self.warmup_steps) / self.decay_steps)
                    ),
                )

        else:
            raise ValueError(f"Unknown lr_scheduler: {self.name}")

    def update_lr(self, optimizer, iteration):
        lr = self.get_lr(iteration)
        for param_group in optimizer.param_groups:  # inplace
            param_group["lr"] = lr

    def get_lrs_list(self):
        return [self.get_lr(i) for i in range(self.max_iterations)]
