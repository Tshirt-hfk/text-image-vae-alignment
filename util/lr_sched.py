"""
Learning rate schedulers with warmup support.
Adapted from JiT util/lr_sched.py for AlignmentVAE.

Supports separate LR scaling per param group (e.g., different LR for VAE
and label encoder).
"""

import math


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup.

    Supports per-iteration calls: epoch can be fractional
    (e.g., data_iter_step / len(data_loader) + epoch_int).

    Each optimizer param group can have an optional 'lr_scale' factor.

    Args:
        optimizer: torch optimizer
        epoch: current epoch (float, supports fractional for per-iter scheduling)
        args: namespace with lr, min_lr, warmup_epochs, epochs, lr_schedule
    """
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    lr_schedule = getattr(args, 'lr_schedule', 'cosine')
    min_lr = getattr(args, 'min_lr', 0.0)
    base_lr = args.lr

    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        if lr_schedule == "constant":
            lr = base_lr
        elif lr_schedule == "cosine":
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise NotImplementedError(f"Unknown lr_schedule: {lr_schedule}")

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
