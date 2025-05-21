import torch
from models.base import BaseModel
from maths import spectral

def compute_stats_layer_linear(layer, prefix=""):
    """Compute statistics and return a dict of 
    named scalar for given Torch linear layer"""
    
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        return {
            prefix+"weight_meanabs": torch.mean(torch.abs(layer.weight)),
            prefix+"bias_meanabs": torch.mean(torch.abs(layer.bias)),
            prefix+"weight_maxabs": torch.max(torch.abs(layer.weight)),
            prefix+"bias_maxabs": torch.max(torch.abs(layer.bias)),
            prefix+"weight_spectral_norm": spectral.spectral_norm(layer.weight),
        }

class MLPNet(BaseModel):
    """Multi layer perception implementation"""

    def __init__(self, dim_in, dim_hidden, dim_out, 
            activation=torch.nn.Tanh, 
            spectral_norm_on_hidden=False,
            norm_layer=False,
        ):
        """Init with dimension for input, list of hidden layers, output and torch activation function.
        spectral_norm_on_hidden: If True, spectral normalization is applied on each hidden layers.""" 
        super(MLPNet, self).__init__()

        #Check arguments
        if not isinstance(dim_hidden, list):
            raise IOError("dim_hidden must be a list")
        if len(dim_hidden) == 0:
            raise IOError("dim_hidden is empty")

        #Save dimensions
        self._dim_in = dim_in
        self._dim_out = dim_out

        #Create torch layers
        self._linear_in = torch.nn.Linear(dim_in, dim_hidden[0])
        self._linear_hidden = torch.nn.ModuleList([])
        for i in range(1, len(dim_hidden)):
            if spectral_norm_on_hidden:
                self._linear_hidden.append(
                    torch.nn.utils.parametrizations.spectral_norm(
                        torch.nn.Linear(dim_hidden[i-1], dim_hidden[i])))
            else:
                self._linear_hidden.append(torch.nn.Linear(dim_hidden[i-1], dim_hidden[i]))
            if norm_layer:
                self._linear_hidden.append(torch.nn.LayerNorm(dim_hidden[i]))
        self._linear_out = torch.nn.Linear(dim_hidden[-1], dim_out)
        self._activation = activation()

        #Singular vector cache for spectral norm computation
        self._singular_vect_in = None
        self._singular_vect_hidden = [None for _ in range(len(self._linear_hidden))]
        self._singular_vect_out = None

    def forward(self, *tensors):
        """Run the MLP with given (optionally several) batched tensors"""

        #Concatenate input argument tensors
        x = torch.cat(tensors, dim=1)
        torch._assert(x.shape[1] == self._dim_in, 
            "Invalid input tensor dimension: " + str(x.shape[1]) + " != " + str(self._dim_in))

        #Run the layers
        x = self._activation(self._linear_in(x))
        for i in range(0, len(self._linear_hidden)):
            x = self._activation(self._linear_hidden[i](x))
        x = self._linear_out(x)
        return x

    def compute_spectral_norm(self, power_iterations=1):
        """Compute and return the spectral norm of the network.
        Use internaly cached singular vector and power iteration to speed up the computation"""

        norm = 1.0
        sigma_in, self._singular_vect_in = spectral.spectral_norm_power_iteration(
            self._linear_in.weight, 
            self._singular_vect_in, 
            iterations=power_iterations)
        norm *= sigma_in
        for i in range(0, len(self._linear_hidden)):
            sigma_hidden, self._singular_vect_hidden[i] = spectral.spectral_norm_power_iteration(
                self._linear_hidden[i].weight, 
                self._singular_vect_hidden[i], 
                iterations=power_iterations)
            norm *= sigma_hidden
        sigma_out, self._singular_vect_out = spectral.spectral_norm_power_iteration(
            self._linear_out.weight, 
            self._singular_vect_out, 
            iterations=power_iterations)
        norm *= sigma_out

        return norm

    def get_params_stats(self):
        """Return a dict of named scalars providing statistics 
        about the current model parameters"""

        lipschitz_constant = 1.0
        stats = {} 
        stats = {**stats, **compute_stats_layer_linear(self._linear_in, "layer_in/")}
        lipschitz_constant *= stats["layer_in/weight_spectral_norm"]
        for i in range(0, len(self._linear_hidden)):
            stats = {**stats, **compute_stats_layer_linear(self._linear_hidden[i], "layer_hidden_"+str(i)+"/")}
            lipschitz_constant *= stats["layer_hidden_"+str(i)+"/weight_spectral_norm"]
        stats = {**stats, **compute_stats_layer_linear(self._linear_out, "layer_out/")}
        lipschitz_constant *= stats["layer_out/weight_spectral_norm"]
        stats["lipschitz_constant"] = lipschitz_constant
        return stats

