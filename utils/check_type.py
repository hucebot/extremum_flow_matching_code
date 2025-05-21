import numpy as np
import torch

def is_scalar(value):
    """Return True if given value is a scalar (float, numpy, torch)"""

    if isinstance(value, float):
        return True
    elif isinstance(value, np.float64):
        return True
    elif isinstance(value, np.float32):
        return True
    elif isinstance(value, np.ndarray) and value.size == 1:
        return True
    elif torch.is_tensor(value) and torch.numel(value) == 1:
        return True
    else:
        return False

def get_scalar(value):
    """Return the float value from given multi-type 
    scalar inputs (float, numpy, torch)"""

    assert is_scalar(value)
    if isinstance(value, np.ndarray):
        return float(value)
    elif torch.is_tensor(value):
        return float(value.detach().numpy())
    else:
        return float(value)

