import torch

class BaseModel(torch.nn.Module):
    """Base class for models"""
    def __init__(self):
        super().__init__()
    def print_parameters(self, 
            print_all: bool = False,
        ):
        """Print model parameters.
        Args:
            print_all: If True, detail all layers.
        """
        print(type(self).__name__)
        sum_params_all = 0
        sum_params_grad = 0
        for name,param in self.named_parameters():
            sum_params_all += param.numel()
            if param.requires_grad:
                sum_params_grad += param.numel()
            if print_all:
                print("    Params:", name, param.numel(), "requires_grad:", param.requires_grad)
        print("    Total all:", sum_params_all, "Total grad:", sum_params_grad)

