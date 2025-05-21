import torch
import numpy as np
import torch.nn.functional as F

def spectral_norm_power_iteration(mat, singular_vect_right=None, iterations=10):
    """Compute the spectral norm i.e. the largest 
    singular values of given 2d matrix.
    singular_vect_right: If not None, use this right singular vector 
    estimation to warm start the power iteration.
    iterations: Number of iteration of the power method to run."""

    #Check inputs
    assert torch.is_tensor(mat) and mat.dim() == 2
    assert singular_vect_right is None or \
        (torch.is_tensor(singular_vect_right) and singular_vect_right.dim() == 1 \
        and singular_vect_right.size()[0] == mat.size()[1])
    assert isinstance(iterations, int) and iterations > 0

    #Random (isotropic) initialization if None is provided
    if singular_vect_right is None:
        singular_vect_right = torch.randn(mat.size()[1])
    #Initial normalization of the singular vector estimation
    singular_vect_right = F.normalize(singular_vect_right, dim=0)
    #Power iterations method
    for _ in range(iterations):
        singular_vect_left = F.normalize(torch.mv(mat, singular_vect_right), dim=0)
        singular_vect_right = F.normalize(torch.mv(mat.t(), singular_vect_left), dim=0)
    #Compute and return the singular value and right vector
    sigma = torch.dot(singular_vect_left, torch.mv(mat, singular_vect_right))

    return sigma, singular_vect_right.detach()

def spectral_norm(mat):
    """Compute the spectral norm as Torch matrix norm"""
    
    #Check inputs
    assert torch.is_tensor(mat) and mat.dim() == 2
    return torch.linalg.norm(mat, ord=2)

