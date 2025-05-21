import torch

def uniform_unit_sphere(
        size_sample: int, 
        size_channel: int,
        device: torch.device = None,
    ) -> torch.Tensor:
    """Sample uniformly on unit sphere.
    Args:
        size_sample: Batch size to sample.
        size_channel: Dimension of the output vector 
            sampled on dimension-sphere.
        device: Optional torch device.
    Returns:
        Sampled batch tensor with unit norm (size_sample, size_channel).
    """
    tmp_tensor = torch.randn(size_sample, size_channel, device=device)
    tmp_norm = torch.linalg.vector_norm(tmp_tensor, keepdim=True, dim=1) 
    return torch.div(tmp_tensor, tmp_norm)

def uniform_unit_ball(
        size_sample: int, 
        size_channel: int,
        device: torch.device = None,
    ) -> torch.Tensor:
    """Sample uniformly within unit ball.
    Args:
        size_sample: Batch size to sample.
        size_channel: Dimension of the output vector 
            sampled in dimension-ball.
        device: Optional torch device.
    Returns:
        Sampled batch tensor with norm 
            lower than one (size_sample, size_channel).
    """
    tmp_tensor = uniform_unit_sphere(size_sample, size_channel, device=device)
    tmp_radius = torch.rand(size_sample, 1, device=device) ** (1.0/size_channel)
    return torch.mul(tmp_tensor, tmp_radius)

