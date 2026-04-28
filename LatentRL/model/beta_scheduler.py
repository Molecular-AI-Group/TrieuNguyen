import numpy as np


class BetaScheduler:
    """
    Beta scheduler for controlling the KL-divergence weight in VAEs.
    Supports linear, sigmoid, step warm-up scheduling, and cyclical annealing.
    """

    def __init__(
        self,
        total_epochs,
        schedule="linear",
        start=0.0,
        end=1.0,
        warmup_fraction=0.5,
        n_cycles=4,
        cycle_schedule="linear",
    ):
        """
        Args:
            total_epochs (int): Total number of training epochs.
            schedule (str): "linear", "sigmoid", "step", or "cyclical".
            start (float): Starting beta value.
            end (float): Final beta value.
            warmup_fraction (float): Fraction of epochs used for warmup (only for linear/sigmoid/step).
            n_cycles (int): Number of annealing cycles (only for cyclical).
            cycle_schedule (str): Schedule within each cycle - "linear", "sigmoid", or "cosine".
        """
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.start = start
        self.end = end
        self.warmup_fraction = warmup_fraction
        self.n_cycles = n_cycles
        self.cycle_schedule = cycle_schedule

    def get_beta(self, epoch):
        """Return beta value for a given epoch."""
        if self.schedule == "linear":
            warmup_epochs = int(self.total_epochs * self.warmup_fraction)
            progress = min(epoch / warmup_epochs, 1.0)
            return self.start + (self.end - self.start) * progress

        elif self.schedule == "sigmoid":
            midpoint = self.total_epochs * self.warmup_fraction
            steepness = 10 / self.total_epochs
            return self.start + (self.end - self.start) / (
                1 + np.exp(-steepness * (epoch - midpoint))
            )

        elif self.schedule == "step":
            warmup_epochs = int(self.total_epochs * self.warmup_fraction)
            return self.start if epoch < warmup_epochs else self.end

        elif self.schedule == "cyclical":
            return self._cyclical_beta(epoch)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")

    def _cyclical_beta(self, epoch):
        """
        Cyclical annealing schedule that periodically resets beta to start value.
        
        Args:
            epoch (int): Current epoch number.
            
        Returns:
            float: Beta value for the current epoch.
        """
        cycle_length = self.total_epochs / self.n_cycles
        cycle_position = epoch % cycle_length
        cycle_progress = cycle_position / cycle_length

        if self.cycle_schedule == "linear":
            beta = self.start + (self.end - self.start) * cycle_progress

        elif self.cycle_schedule == "sigmoid":
            midpoint = 0.5
            steepness = 10
            beta = self.start + (self.end - self.start) / (
                1 + np.exp(-steepness * (cycle_progress - midpoint))
            )

        elif self.cycle_schedule == "cosine":
            # Cosine annealing from start to end
            beta = self.start + (self.end - self.start) * (
                1 - np.cos(np.pi * cycle_progress)
            ) / 2

        else:
            raise ValueError(f"Unknown cycle schedule type: {self.cycle_schedule}")

        return beta


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    total_epochs = 20
    
    # Create different schedulers
    schedulers = {
        "Cyclical (Linear)": BetaScheduler(
            total_epochs, schedule="cyclical", n_cycles=5, cycle_schedule="linear"
        ),



    }

    # Plot beta schedules
    plt.figure(figsize=(12, 6))
    epochs = np.arange(total_epochs)

    for name, scheduler in schedulers.items():
        betas = [scheduler.get_beta(e) for e in epochs]
        plt.plot(epochs, betas, label=name, linewidth=2)

    plt.xlabel("Epoch")
    plt.xticks(epochs)
    plt.ylabel("Beta Value")
    plt.title("Beta Scheduling Strategies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("beta_schedules.png")
    plt.show()