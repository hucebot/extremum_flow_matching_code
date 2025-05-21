import numpy as np
import torch
from typing import List, Union, Dict, Tuple
from models.base import BaseModel
from models.mlp import MLPNet

class ContinuousPositionEmbedding(BaseModel):
    """Sinus positional embedding adapted for continuous signal with given range"""
    def __init__(self, 
            size_emb: int, 
            period_min: float, 
            period_max: float,
        ):
        """Initialization.
        Args:
            size_emb: Total dimension of embedding output, and generate size_emb//2 oscillations.
            period_min: Period of the fastest oscillation.
            period_max: Period of the slowest oscillation.
        """
        super().__init__()
        self._size_emb = size_emb
        #Generate oscillation periods
        size_half = self._size_emb//2
        #(size_half)
        tensor_period_ratio = torch.linspace(0.0, 1.0, size_half)
        self._periods = period_min*torch.pow(period_max/period_min, tensor_period_ratio)
    def forward(self, 
            tensor_time: torch.Tensor,
        ) -> torch.Tensor:
        """Forward.
        Args:
            tensor_time: Input time (size_batch, 1).
        Returns:
            Generated embedding (size_batch, size_emb).
        """
        torch._assert(torch.is_tensor(tensor_time) and tensor_time.dim() == 2 and tensor_time.size()[1] == 1, "")
        if self._periods.get_device() != tensor_time.get_device():
            self._periods = self._periods.to(tensor_time.device)
        size_batch = tensor_time.size()[0]
        #(size_batch, size_emb//2)
        tensor_phase = torch.divide(
            tensor_time.repeat(1, self._size_emb//2),
            self._periods.unsqueeze(0).repeat(size_batch, 1))
        #(size_batch, size_emb//2, 2)
        tensor_value = torch.stack([
            torch.sin(2.0*np.pi*tensor_phase), 
            torch.cos(2.0*np.pi*tensor_phase)
        ], dim=-1)
        #(size_batch, size_emb)
        tensor_value = tensor_value.reshape(size_batch, self._size_emb)
        return tensor_value

class BlockConv1d(BaseModel):
    """1d convolution keeping same temporal length with non-linearity and normalization"""
    def __init__(self,
            size_channel_in: int,
            size_channel_out: int,
            size_kernel: int,
            size_group_norm: int,
        ):
        """Initialization.
        Args:
            size_channel_in: Input feature size.
            size_channel_out: Output feature size.
            size_kernel: Convolution kernel size.
            size_group_norm: Number of groups for group normalization.
        """
        super().__init__()
        self._block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=size_channel_in, 
                out_channels=size_channel_out,
                kernel_size=size_kernel,
                stride=1,
                padding=size_kernel//2, 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode="zeros"),
            torch.nn.GroupNorm(
                num_groups=size_group_norm, 
                num_channels=size_channel_out,
                affine=True),
            torch.nn.SiLU(),
        )
    def forward(self,
            tensor_in: torch.Tensor,
        ) -> torch.Tensor:
        """Forward.
        Args:
            tensor_in: Input tensor (size_batch, size_channel_in, size_length).
        Returns
            Output tensor (size_batch, size_channel_out, size_length).
        """
        return self._block(tensor_in)
class BlockDownsample(BaseModel):
    """Downscale the sequence by 2"""
    def __init__(self,
            size_channel: int,
        ):
        """Initialization.
        Args:
            size_channel: Feature size.
        """
        super().__init__()
        self._conv = torch.nn.Conv1d(
            in_channels=size_channel, 
            out_channels=size_channel,
            kernel_size=2,
            stride=2,
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode="zeros")
    def forward(self, 
            tensor_in: torch.Tensor,
        ) -> torch.Tensor:
        """Forward
        Args:
            tensor_in: Input tensor (size_batch, size_channel, size_length).
        Returns:
            Output tensor (size_batch, size_channel, size_length//2).
        """
        return self._conv(tensor_in)
class BlockUpsample(BaseModel):
    """Upscale the sequence by 2"""
    def __init__(self,
            size_channel: int,
        ):
        """Initialization.
        Args:
            size_channel: Feature size.
        """
        super().__init__()
        self._conv = torch.nn.Conv1d(
            in_channels=size_channel, 
            out_channels=2*size_channel,
            kernel_size=3,
            stride=1,
            padding=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode="zeros")
    def forward(self, 
            tensor_in: torch.Tensor,
        ) -> torch.Tensor:
        """Forward
        Args:
            tensor_in: Input tensor (size_batch, size_channel, size_length).
        Returns:
            Output tensor (size_batch, size_channel, size_length*2).
        """
        tensor_out = self._conv(tensor_in)
        tensor_out = tensor_out.reshape(tensor_in.size()[0], tensor_out.size()[1]//2, tensor_out.size()[2]*2)
        return tensor_out
class BlockConv1dResidualConditional(BaseModel):
    """Convolutional block with residual connection and conditioning using FiLM"""
    def __init__(self, 
            size_channel_in: int,
            size_channel_out: int,
            size_cond: int,
            size_kernel: int,
            size_group_norm: int,
        ):
        """Initialization.
        Args:
            size_channel_in: Input feature size.
            size_channel_out: Output feature size.
            size_cond: Conditioning feature size.
            size_kernel: Convolution kernel size.
            size_group_norm: Number of group for group normalization.
        """
        super().__init__()
        self._block_conv1 = BlockConv1d(
            size_channel_in,
            size_channel_out,
            size_kernel,
            size_group_norm)
        self._block_conv2 = BlockConv1d(
            size_channel_out,
            size_channel_out,
            size_kernel,
            size_group_norm)
        self._conv_residual = torch.nn.Conv1d(
            in_channels=size_channel_in, 
            out_channels=size_channel_out,
            kernel_size=1,
            stride=1,
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True,
            padding_mode="zeros")
        self._cond_encoder = torch.nn.Linear(
            size_cond, 
            2*size_channel_out)
    def forward(self, 
            tensor_in: torch.Tensor, 
            tensor_cond: torch.Tensor,
        ) -> torch.Tensor:
        """Forward.
        Args:
            tensor_in: Input feature tensor (size_batch, size_channel_in, size_length).
            tensor_cond: Conditioning tensor (size_batch, size_cond).
        Returns:
            Output tensor (size_batch, size_channel_out, size_length).
        """
        #Apply conditioning encoder (size_batch, 2*size_cond)
        tensor_cond_film = self._cond_encoder(tensor_cond)
        #Retrieve FiLM scale and bias components
        tensor_cond_film = tensor_cond_film.reshape(tensor_cond_film.size()[0], tensor_cond_film.size()[1]//2, 2)
        #(size_batch, size_cond)
        tensor_cond_scale = tensor_cond_film[:,:,0]
        tensor_cond_bias = tensor_cond_film[:,:,1]
        #Expend dim for broadcasting (size_batch, size_cond, 1)
        tensor_cond_scale = tensor_cond_scale.unsqueeze(-1)
        tensor_cond_bias = tensor_cond_bias.unsqueeze(-1)
        #Apply first convolution
        tensor_out = self._block_conv1(tensor_in)
        #Apply FiLM conditioning
        tensor_out = tensor_cond_scale*tensor_out + tensor_cond_bias
        #Apply second convolution
        tensor_out = self._block_conv2(tensor_out)
        ##Add residual connection
        tensor_out = tensor_out + self._conv_residual(tensor_in)
        return tensor_out
class ModelUnetResidualConditional(BaseModel):
    """Implement a 1d convolutional Unet architecture with 
    residual block, group normalization and conditional vector.
    Uses position sinusoidal embedding.
    """
    def __init__(self,
            size_channel: int,
            size_emb_transport: int,
            size_cond: int,
            size_channel_hidden: List[int],
            period_min: float, 
            period_max: float,
            size_kernel: int,
            size_group_norm:int,
        ):
        """Initialization.
        Args:
            size_channel: Input trajectory feature size.
            size_emb_transport: Total dimension of 
                transport ratio embedding.
            size_cond: Input conditioning vector size.
            size_channel_hidden: List of intermediate feature size.
                The length of this list define the number of levels.
            period_min: Period of fastest oscillation
                of transport ratio embedding.
            period_max: Period of slowest oscillation
                of transport ratio embedding.
            size_kernel: Convolution kernel size.
            size_group_norm: Number of groups for
                group normalization.
        """
        super().__init__()
        #Transport ratio embedding and encoder
        self._transport_encoder = torch.nn.Sequential(
            ContinuousPositionEmbedding(
                size_emb_transport,
                period_min,
                period_max),
            torch.nn.Linear(
                size_emb_transport,
                size_emb_transport*4),
            torch.nn.SiLU(),
            torch.nn.Linear(
                size_emb_transport*4,
                size_emb_transport),
            torch.nn.SiLU(),
        )
        size_cond_all = size_emb_transport + size_cond
        #Process feature sizes
        list_size_channel = [size_channel] + size_channel_hidden
        size_channel_last = list_size_channel[-1]
        #Downsample path
        self._module_down = torch.nn.ModuleList([])
        for i in range(len(list_size_channel)-1):
            tmp_size_in = list_size_channel[i]
            tmp_size_out = list_size_channel[i+1]
            self._module_down.append(torch.nn.Sequential(
                BlockConv1dResidualConditional(
                    size_channel_in=tmp_size_in,
                    size_channel_out=tmp_size_out,
                    size_cond=size_cond_all,
                    size_kernel=size_kernel,
                    size_group_norm=size_group_norm),
                BlockConv1dResidualConditional(
                    size_channel_in=tmp_size_out,
                    size_channel_out=tmp_size_out,
                    size_cond=size_cond_all,
                    size_kernel=size_kernel,
                    size_group_norm=size_group_norm),
                BlockDownsample(
                    size_channel=tmp_size_out),
            ))
        #Middle path
        self._module_middle = torch.nn.Sequential(
            BlockConv1dResidualConditional(
                size_channel_in=size_channel_last,
                size_channel_out=size_channel_last,
                size_cond=size_cond_all,
                size_kernel=size_kernel,
                size_group_norm=size_group_norm),
            BlockConv1dResidualConditional(
                size_channel_in=size_channel_last,
                size_channel_out=size_channel_last,
                size_cond=size_cond_all,
                size_kernel=size_kernel,
                size_group_norm=size_group_norm),
        )
        #Upsample path taking as input the residual part from
        #the same level as input channels concatenation.
        self._module_up = torch.nn.ModuleList([])
        for i in range(len(list_size_channel)-1, 0, -1):
            tmp_size_in = list_size_channel[i]
            tmp_size_out = list_size_channel[i-1]
            #Output channel size of the last residual convolution 
            #of the upsample path is set to repeat first hidden size 
            #such that the final layer can perform the last transformation
            if i == 1:
                tmp_size_out = size_channel_hidden[0]
            self._module_up.append(torch.nn.Sequential(
                BlockUpsample(
                    size_channel=tmp_size_in),
                BlockConv1dResidualConditional(
                    size_channel_in=2*tmp_size_in,
                    size_channel_out=tmp_size_out,
                    size_cond=size_cond_all,
                    size_kernel=size_kernel,
                    size_group_norm=size_group_norm),
                BlockConv1dResidualConditional(
                    size_channel_in=tmp_size_out,
                    size_channel_out=tmp_size_out,
                    size_cond=size_cond_all,
                    size_kernel=size_kernel,
                    size_group_norm=size_group_norm),
            ))
        #Final convolution layer
        self._conv_final = torch.nn.Conv1d(
            in_channels=size_channel_hidden[0],
            out_channels=size_channel,
            kernel_size=1,
            stride=1,
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode="zeros")
    def forward(self, 
            tensor_traj: torch.Tensor,
            tensor_transport: torch.Tensor,
            tensor_cond: Union[torch.Tensor, None] = None,
        ) -> torch.Tensor:
        """Forward.
        Args:
            tensor_traj: Input trajectory (size_batch, size_length, size_channel).
            tensor_transport: Input transport ratio in [0:1] (size_batch, 1).
            tensor_cond: Input conditional vector (size_batch, size_cond) or None.
        Returns:
            Output trajectory (size_batch, size_length, size_channel).
        """
        #Transport ratio embedding
        #(size_batch, size_emb_transport)
        tensor_transport_embedded = self._transport_encoder(tensor_transport)
        #Concat external conditioning to generate full conditioning tensor
        #(size_batch, size_cond_all)
        if tensor_cond is not None:
            tensor_cond_all = torch.cat([tensor_transport_embedded, tensor_cond], dim=1)
        else:
            tensor_cond_all = tensor_transport_embedded
        #Convert tensor axis order
        #(size_batch, size_length, size_channel)
        tensor_out = tensor_traj
        tensor_out = tensor_out.transpose(1, 2)
        #Downsample path
        list_residuals = []
        for i in range(len(self._module_down)):
            (netResConv1, netResConv2, netDown) = self._module_down[i]
            tensor_out = netResConv1(tensor_out, tensor_cond_all)
            tensor_out = netResConv2(tensor_out, tensor_cond_all)
            list_residuals.append(tensor_out)
            tensor_out = netDown(tensor_out)
        #Middle path
        (netResConv1, netResConv2) = self._module_middle
        tensor_out = netResConv1(tensor_out, tensor_cond_all)
        tensor_out = netResConv2(tensor_out, tensor_cond_all)
        #Upsample path
        for i in range(len(self._module_up)):
            (netUp, netResConv1, netResConv2) = self._module_up[i]
            tensor_out = netUp(tensor_out)
            tensor_out = torch.cat(
                [tensor_out, list_residuals[len(list_residuals)-1-i]], 
                dim=1)
            tensor_out = netResConv1(tensor_out, tensor_cond_all)
            tensor_out = netResConv2(tensor_out, tensor_cond_all)
        #Last convolution layer
        tensor_out = self._conv_final(tensor_out)
        #Reconvert tensor axis order
        #(size_batch, size_length, size_channel)
        tensor_out = tensor_out.transpose(1, 2)
        return tensor_out

class ModelDenseSimple(BaseModel):
    """Simple MLP model concatenating all features as input layer"""
    def __init__(self, 
            size_channel: int,
            size_length: int,
            size_cond: int,
            size_hidden_list: List[int],
        ):
        """Initialization.
            size_channel: Input trajectory feature size.
            size_length: Input trajectory spatial size.
            size_cond: Input conditioning vector size.
            size_hidden_list: List of hidden layers size.
        Args:
        """
        super().__init__()
        size_in = size_channel*size_length + 1 + size_cond
        self._mlp = MLPNet(
            size_in, size_hidden_list, size_channel*size_length,
            activation=torch.nn.SiLU, 
            spectral_norm_on_hidden=False,
            norm_layer=True)
        self._size_length = size_length
        self._size_channel = size_channel
    def forward(self, 
            tensor_traj: torch.Tensor,
            tensor_transport: torch.Tensor,
            tensor_cond: Union[torch.Tensor, None] = None,
        ) -> torch.Tensor:
        """Forward.
        Args:
            tensor_traj: Input trajectory (size_batch, size_length, size_channel).
            tensor_transport: Input transport ratio in [0:1] (size_batch, 1).
            tensor_cond: Input conditional vector (size_batch, size_cond) or None.
        Returns:
            Output trajectory (size_batch, size_length, size_channel).
        """
        if tensor_cond is not None:
            tensor_in = torch.cat(
                [tensor_traj.flatten(1), tensor_transport, tensor_cond],
                dim=1)
        else:
            tensor_in = torch.cat(
                [tensor_traj.flatten(1), tensor_transport],
                dim=1)
        tensor_out = self._mlp(tensor_in)
        tensor_out = tensor_out.reshape(
            tensor_out.size()[0], 
            self._size_length, 
            self._size_channel)
        return tensor_out

class ModelDenseEmbedding(BaseModel):
    """MLP model concatenating all features as input layers 
    and using ContinuousPositionEmbedding"""
    def __init__(self, 
            size_channel: int,
            size_length: int,
            size_cond: int,
            size_hidden_list: List[int],
            size_emb_transport: int,
            period_min: float, 
            period_max: float,
        ):
        """Initialization.
            size_channel: Input trajectory feature size.
            size_length: Input trajectory spatial size.
            size_cond: Input conditioning vector size.
            size_hidden_list: List of hidden layers size.
            size_emb_transport: Total dimension of 
                transport ratio embedding.
            period_min: Period of fastest oscillation
                of transport ratio embedding.
            period_max: Period of slowest oscillation
                of transport ratio embedding.
        Args:
        """
        super().__init__()
        size_in = size_channel*size_length + size_emb_transport + size_cond
        self._emb = ContinuousPositionEmbedding(
            size_emb_transport,
            period_min,
            period_max)
        self._mlp = MLPNet(
            size_in, size_hidden_list, size_channel*size_length,
            activation=torch.nn.SiLU, spectral_norm_on_hidden=False)
        self._size_length = size_length
        self._size_channel = size_channel
    def forward(self, 
            tensor_traj: torch.Tensor,
            tensor_transport: torch.Tensor,
            tensor_cond: Union[torch.Tensor, None] = None,
        ) -> torch.Tensor:
        """Forward.
        Args:
            tensor_traj: Input trajectory (size_batch, size_length, size_channel).
            tensor_transport: Input transport ratio in [0:1] (size_batch, 1).
            tensor_cond: Input conditional vector (size_batch, size_cond) or None.
        Returns:
            Output trajectory (size_batch, size_length, size_channel).
        """
        tensor_emb = self._emb(tensor_transport)
        if tensor_cond is not None:
            tensor_in = torch.cat(
                [tensor_traj.flatten(1), tensor_emb, tensor_cond],
                dim=1)
        else:
            tensor_in = torch.cat(
                [tensor_traj.flatten(1), tensor_emb],
                dim=1)
        tensor_out = self._mlp(tensor_in)
        tensor_out = tensor_out.reshape(
            tensor_out.size()[0], 
            self._size_length, 
            self._size_channel)
        return tensor_out

