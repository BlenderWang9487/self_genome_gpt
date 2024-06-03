import math


def get_cosine_schedule_with_warmup_and_min_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    lr: float,
    min_lr: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    lr_scale = 0.5 * (
        1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
    )  # (0, 1)

    # should rescale to be (min_lr / lr, 1)
    lower_ratio = float(min_lr / lr)
    lr_scale = lower_ratio + (1 - lower_ratio) * lr_scale
    return max(0.0, lr_scale)
