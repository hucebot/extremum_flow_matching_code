import torch
from typing import List, Union, Callable, Tuple

class FlowTransport():
    """Implementation of rectify flow for transport of distribution"""
    def __init__(self, 
            type_transport_func: str,
        ):
        """Initialization.
        Args:
            type_transport_func: Tranformation function type 
                for transport ratio.
        """
        if type_transport_func == "linear":
            self._func_transport = FlowTransport.func_transport_ratio_linear
        elif type_transport_func == "cosine_1":
            self._func_transport = FlowTransport.func_transport_ratio_cosine_1
        elif type_transport_func == "cosine_2":
            self._func_transport = FlowTransport.func_transport_ratio_cosine_2
        else:
            raise IOError("Invalid type:" + type_transport_func);
    @staticmethod
    def func_transport_ratio_linear(
            ratio: torch.Tensor,
        ) -> torch.Tensor:
        """Linear transformation for transport ratio function"""
        return ratio
    @staticmethod
    def func_transport_ratio_cosine_1(
            ratio: torch.Tensor,
        ) -> torch.Tensor:
        """Cosine transformation for transport ratio function"""
        return 0.5*(1.0-torch.cos(np.pi*ratio))
    @staticmethod
    def func_transport_ratio_cosine_2(
            ratio: torch.Tensor,
        ) -> torch.Tensor:
        """Double cosine transformation for transport ratio function"""
        ratio = 0.5*(1.0-torch.cos(np.pi*ratio))
        ratio = 0.5*(1.0-torch.cos(np.pi*ratio))
        return ratio
    def transport_interpolate(self,
            tensor_src: torch.Tensor,
            tensor_dst: torch.Tensor,
            tensor_ratio: torch.Tensor,
        ) -> torch.Tensor:
        """Perform transport interpolation
        Args:
            tensor_src: Source samples (size_batch, size_length, size_channel).
            tensor_dst: Target samples (size_batch, size_length, size_channel).
            tensor_ratio: Transport ratio in [0:1] (size_batch, 1).
        Returns:
            Interpolated samples between source and 
                target according to given ratio (size_batch, size_length, size_channel).
        """
        #(size_batch, 1)
        alpha = self._func_transport(tensor_ratio)
        #(size_batch, 1, 1) to allow broadcasting
        alpha = alpha.unsqueeze(2)
        return (1.0-alpha)*tensor_src + alpha*tensor_dst
    def transport_forward(self,
            model: torch.nn.Module, 
            traj_src: torch.Tensor, 
            tensor_cond: Union[torch.Tensor, None],
            steps: Union[int, torch.Tensor],
            func_postprocess: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
        ) -> List[torch.Tensor]:
        """Compute forward flow transportation from source to target distribution.
        Args:
            model: Model computing transportation step.
            traj_src: Initial source trajectory sample (size_batch, size_length, size_channel).
            tensor_cond: Optional conditioning tensor (size_batch, size_cond), or None.
            steps: If is an integer, the number of transport steps. 
                If is a 1d tensor (size_batch, steps+1), provide non uniformly spaced 
                steps between 0.0 and 1.0.
            func_postprocess: Callback function used to post-process and update 
                the batched trajectory (size_batch, size_length, size_channel) 
                at the end of each transport step.
        Returns:
            The chain of all intermediate trajectory tensors 
            (size_batch, size_length, size_channel) in a list of length steps.
        """
        chain = []
        size_batch = traj_src.size()[0]
        model.eval()
        #Generate transport ratio steps
        if isinstance(steps, int):
            ratio_steps = torch.linspace(0.0, 1.0, steps+1, device=traj_src.device) \
                .unsqueeze(0).repeat(size_batch,1)
        else:
            torch._assert(torch.is_tensor(steps) and steps.dim() == 2, "")
            torch._assert(steps.size(0) == size_batch and steps.size(1) >= 2, "")
            ratio_steps = steps
        #Transport loop
        chain.append(traj_src.clone())
        for k in range(ratio_steps.size(1)-1):
            #(size_batch, 1)
            ratio = ratio_steps[:,k].unsqueeze(1)
            #(size_batch, 1, 1) extend size for broadcasting
            alpha_1 = self._func_transport(
                ratio_steps[:,k].unsqueeze(1).unsqueeze(2))
            alpha_2 = self._func_transport(
                ratio_steps[:,k+1].unsqueeze(1).unsqueeze(2))
            #Evaluate the model
            delta = model(traj_src, ratio, tensor_cond)
            #Apply flow transport step
            traj_src = traj_src + (alpha_2-alpha_1)*delta
            #Call post process callback
            if func_postprocess is not None:
                traj_src = func_postprocess(traj_src)
            chain.append(traj_src.clone())
        return chain
    def transport_backward(self,
            model: torch.nn.Module, 
            traj_dst: torch.Tensor, 
            tensor_cond: Union[torch.Tensor, None],
            steps: int,
            func_postprocess: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
        ) -> List[torch.Tensor]:
        """Compute backward flow transportation from target to source distribution.
        Args:
            model: Model computing transportation step.
            traj_dst: Initial target trajectory sample (size_batch, size_length, size_channel).
            tensor_cond: Optional conditioning tensor (size_batch, size_cond), or None.
            steps: The number of transport steps. 
            func_postprocess: Callback function used to post-process and update 
                the batched trajectory (size_batch, size_length, size_channel) 
                at the end of each transport step.
        Returns:
            The chain of all intermediate trajectory tensors 
            (size_batch, size_length, size_channel) in a list of length steps.
        """
        size_batch = traj_dst.size()[0]
        assert isinstance(steps, int)
        ratio_steps = torch.linspace(1.0, 0.0, steps+1, device=traj_dst.device) \
            .unsqueeze(0).repeat(size_batch,1)
        return self.transport_forward(
            model, traj_dst, tensor_cond, ratio_steps, func_postprocess)
    def train_loss(self,
            model: torch.nn.Module,
            traj_src: torch.Tensor,
            tensor_cond: Union[torch.Tensor, None],
            traj_dst: torch.Tensor,
            weights: Union[torch.Tensor, None] = None,
        ) -> torch.Tensor:
        """Compute transport flow training loss.
        Args:
            model: Model to train computing transportation  step.
            traj_src: Source trajectory samples (size_batch, size_length, size_channel).
            tensor_cond: Optional conditioning tensor (size_batch, size_cond), or None.
            traj_dst: Target trajectory samples (size_batch, size_length, size_channel).
            weights: Optional weights applied to batched loss errors (size_batch).
        Returns:
            Scalar loss tensor (1).
        """

        size_batch = traj_src.size()[0]
        torch._assert(weights is None or (len(weights.size()) == 1 and weights.size(0) == size_batch), "")
        if weights is None:
            weights = torch.ones(size_batch, device=traj_dst.device)
        weights = weights.unsqueeze(1).unsqueeze(2)
        tensor_ratio = torch.rand((size_batch, 1), device=traj_src.device)
        model.train()
        traj_blend = self.transport_interpolate(
            traj_src, traj_dst, tensor_ratio)
        traj_delta = model(traj_blend, tensor_ratio, tensor_cond)
        loss = (weights*(traj_delta-(traj_dst-traj_src)).pow(2)).mean()
        return loss
    def divergence(self,
            model: torch.nn.Module,
            trajs: torch.Tensor,
            tensor_cond: Union[torch.Tensor, None],
            tensor_ratio: torch.Tensor,
            method: str = "jacobian1",
            create_graph: bool = False,
        ) -> torch.Tensor:
        """Compute batched divergence of the given flow model 
        at given input and transport ratio.
        Args:
            model: Model computing transportation step.
            trajs: Trajectory point on which to evaluate 
                the flow divergence (size_batch, size_length, size_channel).
            tensor_ratio: Transport ratio at which evaluate 
                the flow divergence between 0 and 1 (size_batch, 1).
            tensor_cond: Optional conditioning tensor (size_batch, size_cond), or None.
            method: Computation method option in [numeric, jacobian1, jacobian2].
            create_graph: If False, create graph is disable in Jacobian calculation.
        Returns:
            Computed divergence tensor (size_batch, 1).
        """
        with torch.set_grad_enabled(create_graph):
            torch._assert(len(trajs.size()) == 3 and len(tensor_ratio.size()) == 2, "")
            torch._assert(trajs.size(0) == tensor_ratio.size(0), "")
            size_batch = trajs.size(0)
            size_length = trajs.size(1)
            size_channel = trajs.size(2)
            if method == "numeric":
                tensor_div = torch.zeros(size_batch, 1, device=trajs.device)
                eps = 1e-6
                for i in range(size_length):
                    for j in range(size_channel):
                        trajs1 = trajs.clone()
                        trajs2 = trajs.clone()
                        trajs1[:,i,j] -= eps
                        trajs2[:,i,j] += eps
                        delta1 = model(trajs1, tensor_ratio, tensor_cond)
                        delta2 = model(trajs2, tensor_ratio, tensor_cond)
                        tensor_div += ((delta2[:,i,j]-delta1[:,i,j])/(2.0*eps)).unsqueeze(1)
            elif method == "jacobian1":
                def func_tmp(tensor_in):
                    delta = model(tensor_in, tensor_ratio, tensor_cond)
                    return torch.sum(delta, dim=0)
                tensor_gradient = torch.autograd.functional.jacobian(
                    func_tmp, trajs, create_graph=create_graph)
                tensor_div = torch.einsum("ijbij->b", tensor_gradient).unsqueeze(1)
            elif method == "jacobian2":
                tensor_div = torch.zeros(size_batch, 1, device=trajs.device)
                for i in range(size_length):
                    for j in range(size_channel):
                        def func_tmp(tensor_in):
                            tmp_trajs = trajs.clone()
                            tmp_trajs[:,i,j] = tensor_in
                            delta = model(tmp_trajs, tensor_ratio, tensor_cond)
                            return torch.sum(delta[:,i,j], dim=0)
                        tensor_gradient = torch.autograd.functional.jacobian(
                            func_tmp, trajs[:,i,j], create_graph=create_graph)
                        tensor_div += tensor_gradient.unsqueeze(1)
            else:
                raise IOError("Invalid method")
            torch._assert(len(tensor_div.size()) == 2 and tensor_div.size(0) == size_batch, "")
        return tensor_div
    def likelihood_backward(self,
            model: torch.nn.Module, 
            traj_dst: torch.Tensor, 
            tensor_cond: Union[torch.Tensor, None],
            steps: int,
            func_postprocess: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
            create_graph: bool = False,
        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compute both the flow backward transport from target to source distribution and evaluate
        the flow divergence along the integrated flow line. 
        The integrated divergence is used tom estimage the target sample log probability.
        Args:
            model: Model computing transportation step.
            traj_dst: Initial target trajectory sample (size_batch, size_length, size_channel).
            tensor_cond: Optional conditioning tensor (size_batch, size_cond), or None.
            steps: The number of transport steps. 
            func_postprocess: Callback function used to post-process and update 
                the batched trajectory (size_batch, size_length, size_channel) 
                at the end of each transport step.
            create_graph: If False, create graph is disable in Jacobian calculation.
        Returns:
            chain_state: List of all intermediate trajectory tensors (size_batch, size_length, size_channel).
            chain_div: All intermediate flow divergence integration (steps+1, size_batch, 1).
        """
        with torch.set_grad_enabled(create_graph):
            #Compute backward flow transport integration
            size_batch = traj_dst.size()[0]
            assert isinstance(steps, int)
            ratio_steps = torch.linspace(1.0, 0.0, steps+1, device=traj_dst.device) \
                .unsqueeze(0).repeat(size_batch,1)
            chain = self.transport_forward(
                model, traj_dst, tensor_cond, ratio_steps, func_postprocess)
            #Evaluate the divergence at each intermediate point along the flow in batch
            tmp_trajs = torch.cat(chain[0:steps], dim=0)
            tmp_ratio1 = ratio_steps[:,0:steps].permute(1,0).reshape(size_batch*steps).unsqueeze(1)
            tmp_ratio2 = ratio_steps[:,1:steps+1].permute(1,0).reshape(size_batch*steps).unsqueeze(1)
            tmp_delta = self._func_transport(tmp_ratio2) - self._func_transport(tmp_ratio1)
            tmp_cond = None
            if tensor_cond is not None:
                tmp_cond = tensor_cond.repeat(steps,1)
            tmp_div = self.divergence(
                model, tmp_trajs, tmp_cond, tmp_ratio1, create_graph=create_graph)
            #Integrate the divergence along the flow
            tmp_div = (tmp_div*tmp_delta).reshape(steps, size_batch, 1)
            tmp_div = torch.cat([torch.zeros_like(tmp_div[0:1]), tmp_div], dim=0)
            tmp_div = tmp_div.cumsum(dim=0)
            #Return both the chain of state and divergence
            return chain, tmp_div

def random_transport_steps(
        size_batch: int,
        steps: int,
        device: torch.device = None,
    ) -> torch.Tensor:
    """Return a random linspace of transport step ratios.
    Args:
        size_batch: Batch size.
        steps: Number of transport steps.
        device: Optional allocation device for output tensor.
    Returns:
        Tensor of ratio (size_batch, steps) uniformly sampled 
        around linspace(0,1,steps) from 0.0 to 1.0.
    """
    ratios = torch.linspace(0.0, 1.0, steps+1, device=device).unsqueeze(0).repeat(size_batch,1)
    ratios[:,1:-1] += 0.9*(torch.rand(size_batch,steps-1, device=device)-0.5)/(steps-1)
    return ratios

@torch.no_grad()
def generate_dist_src(
        size_batch: int, 
        size_length: int,
        size_channel: int,
        device: torch.device = None,
    ) -> torch.Tensor:
    """Sample trajectories from source noise distribution
    Returns: 
        Source trajectory (size_batch, size_length, size_channel).
    """
    return 0.4*torch.randn(size_batch, size_length, size_channel, device=device)

