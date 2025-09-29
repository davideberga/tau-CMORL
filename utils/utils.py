import math

import numpy as np

import torch


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def normal_log_density_tanh(x, mean, log_std, std, squashed=False, epsilon=1e-6):
    var = std.pow(2)
    log_prob = -0.5 * ((x - mean) ** 2 / var + 2 * log_std + math.log(2 * math.pi))
    log_prob = log_prob.sum(1, keepdim=True)

    if squashed:
        # Undo tanh to get the pre-squash action (used only for log-prob correction)
        u = torch.atanh(torch.clamp(x, -1 + epsilon, 1 - epsilon))
        # Jacobian correction term: log(1 - tanh(u)^2)
        correction = torch.log(1 - x.pow(2) + epsilon)
        log_prob -= correction.sum(1, keepdim=True)

    return log_prob


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return param


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad
