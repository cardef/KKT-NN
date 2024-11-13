# utils.py
import torch

def transform_variables(variable_values, variables_info, mode='normalize'):
    """
    Normalize or denormalize variables based on the mode.

    Parameters:
    - variable_values (dict): Dictionary of variables with their values.
    - variables_info (dict): Dictionary containing variables' bounds and dependencies.
    - mode (str): 'normalize' or 'denormalize'.

    Returns:
    - transformed_vars (dict): Dictionary of transformed variables.
    """
    transformed_vars = {}
    params = {}
    for name in variables_info:
        var_info = variables_info[name]
        x_min, x_max = var_info['bounds']
        x_min = x_min if not callable(x_min) else x_min(params)
        x_max = x_max if not callable(x_max) else x_max(params)
        var_value = variable_values[name]
        if mode == 'normalize':
            transformed_var = 2 * (var_value - x_min) / (x_max - x_min) - 1
        elif mode == 'denormalize':
            transformed_var = (var_value + 1) * (x_max - x_min) / 2 + x_min
        else:
            raise ValueError("Mode must be 'normalize' or 'denormalize'")
        transformed_vars[name] = transformed_var
        params[name] = transformed_var if mode == 'denormalize' else var_value
    return transformed_vars
