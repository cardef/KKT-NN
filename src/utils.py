# src/utils.py

import torch

def transform_variables(variables, variables_info, mode='normalize'):
    """
    Transforms variables by normalizing or denormalizing them.

    Args:
        variables (dict): Dictionary of variables.
        variables_info (dict): Dictionary containing variable information (dimensions, bounds).
        mode (str): Transformation mode, 'normalize' or 'denormalize'.

    Returns:
        dict: Dictionary of transformed variables.
    """
    transformed = {}
    for name, tensor in variables.items():
        bounds = variables_info[name]['bounds']
        if mode == 'normalize':
            x_min, x_max = bounds
            # Handle bounds if they are callable
            if callable(x_min):
                x_min = x_min(variables)
            if callable(x_max):
                x_max = x_max(variables)
            x_min = x_min if isinstance(x_min, torch.Tensor) else torch.tensor([x_min], device=tensor.device)
            x_max = x_max if isinstance(x_max, torch.Tensor) else torch.tensor([x_max], device=tensor.device)
            transformed[name] = 2 * (tensor - x_min) / (x_max - x_min) - 1  # Normalize to [-1, 1]
        elif mode == 'denormalize':
            x_min, x_max = bounds
            if callable(x_min):
                x_min = x_min(variables)
            if callable(x_max):
                x_max = x_max(variables)
            x_min = x_min if isinstance(x_min, torch.Tensor) else torch.tensor([x_min], device=tensor.device)
            x_max = x_max if isinstance(x_max, torch.Tensor) else torch.tensor([x_max], device=tensor.device)
            transformed[name] = (tensor + 1) * (x_max - x_min) / 2 + x_min  # Denormalize
        else:
            raise ValueError("Mode must be 'normalize' or 'denormalize'")
    return transformed
