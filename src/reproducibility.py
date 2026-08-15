import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy and PyTorch random seeds."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )


def seed_worker(worker_id: int) -> None:
    """Seed an individual PyTorch DataLoader worker."""

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a deterministically seeded PyTorch generator."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
