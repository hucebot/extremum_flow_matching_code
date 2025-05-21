import torch
import numpy as np
import copy

class EMAModel(torch.nn.Module):
    """Implement Exponential Moving Average to lowpass filter given model parameters"""
    def __init__(self, 
            cutoff_period: int,
            warmup_steps: int,
        ):
        """Initialization.
        Args:
            cutoff_period: Cutoff period for lowpass parameter filter.
            warmup_steps: Number of update used for linear warmup 
                that ramps up the cutoff period.
        """
        super().__init__()
        #Reset internal state
        self.reset()
        self._cutoff_period = cutoff_period
        self._warmup_steps = warmup_steps
    def reset(self, 
            model: torch.nn.Module = None,
        ):
        """Reset internal state"""
        self._model = None
        self._iteration = 1
        if model is not None:
            self._model = copy.deepcopy(model)
            #Disable all internal model gradients
            for param in self._model.parameters():
                param.requires_grad = False
    def getModel(self):
        """Returns internal filtered model"""
        return self._model
    def getAlpha(self) -> float:
        """Compute and return alpha coefficient for filtering"""
        if self._iteration < self._warmup_steps:
            tmp_period = self._cutoff_period*(self._iteration/self._warmup_steps)
        else:
            tmp_period = self._cutoff_period
        omega = 2.0*np.pi/tmp_period
        coeff = (1.0-omega/2.0)/(1.0+omega/2.0);
        return np.clip(coeff, 0.0, 1.0)
    @torch.no_grad()
    def update(self,
            model: torch.nn.Module,
        ):
        """Update model filtering with given target model"""
        #Lazy initialization
        if self._model is None:
            self.reset(model)
        #Compute filtering coefficient
        alpha = self.getAlpha()
        #Update state
        self._iteration += 1
        #Apply filtering
        for module, ema_module in zip(model.modules(), self._model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(param, dict):
                    raise IOError("Dict parameter not supported")
                if param.requires_grad:
                    ema_param.mul_(alpha)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=(1.0-alpha))
                else:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
    def forward(self, *args, **kwargs):
        """Forward call to internal filtered model"""
        return self.getModel()(*args, **kwargs)

def scheduler_lr_cosine_warmstart_retry(
        epoch: int, 
        steps_epoch: int, 
        steps_warmstart: int, 
        retry: int,
    ) -> float:
    """Implement cosine learning rate scheduler with warm start and restart.
    Args:
        epoch: Input epoch index.
        steps_epoch: Total number of epoch.
        steps_warmstart: Number of epoch steps for linear warm start.
        retry: Number of retries.
    Returns:
        Learinig rate scaling in [0:1].
    """
    steps_try = steps_epoch//retry
    epoch = epoch%steps_try
    if epoch <= steps_warmstart:
        return epoch/steps_warmstart
    else:
        return 0.5*(1.0+np.cos(np.pi*(epoch-steps_warmstart)/(steps_try-steps_warmstart)))

