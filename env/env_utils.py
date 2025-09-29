

import torch


def check_path_collision(obs, start, end, device):
    for poly in obs[1:]:
        if in_poly(start, end, poly, device):
            return True
    return False


def in_poly(xy0, xy1, poly, device):
    n_pts = 1000
    ts = torch.linspace(0, 1, n_pts).to(device)
    xys = xy0.unsqueeze(0) + (xy1 - xy0).unsqueeze(0) * ts.unsqueeze(1)
    xmin, xmax, ymin, ymax = (
        torch.min(poly[:, 0]),
        torch.max(poly[:, 0]),
        torch.min(poly[:, 1]),
        torch.max(poly[:, 1]),
    )

    inside = torch.logical_and(
        (xys[:, 0] - xmin) * (xmax - xys[:, 0]) >= 0,
        (xys[:, 1] - ymin) * (ymax - xys[:, 1]) >= 0,
    )

    res = torch.any(inside)
    return res


import torch
import numpy as np
import random
import os

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()


def soft_step(x):
    return (torch.tanh(500 * x) + 1)/2

def soft_step_hard(x):
    hard = (x>=0).float()
    soft = (torch.tanh(500 * x) + 1)/2
    return soft + (hard - soft).detach()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: for PyTorch >= 1.8, for stricter determinism
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Not available in older versions