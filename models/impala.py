import torch
from models.base import BaseModel
from models.mlp import MLPNet
from typing import List, Union

#Pytorch implementation of Impala model, inspired by:
#https://github.com/seohongpark/ogbench/blob/master/impls/utils/encoders.py
#https://github.com/openai/coinrun/blob/b73a2d79b8c03b8d3476bf9d74b7ce8b15e2e606/coinrun/policies.py
#https://github.com/LukeDitria/pytorch_tutorials/blob/main/section11_rl/solutions/Procgen_PPO.ipynb
#https://arxiv.org/pdf/1802.01561

class ResnetBlock(BaseModel):
    def __init__(self, 
            size_channel: int,
        ):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(
            in_channels=size_channel, 
            out_channels=size_channel,
            kernel_size=3,
            stride=1,
            padding="same", 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode="zeros")
        self._conv2 = torch.nn.Conv2d(
            in_channels=size_channel, 
            out_channels=size_channel,
            kernel_size=3,
            stride=1,
            padding="same", 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode="zeros")
        torch.nn.init.xavier_uniform_(self._conv1.weight)
        torch.nn.init.xavier_uniform_(self._conv2.weight)
    def forward(self, 
            tensor_in: torch.Tensor,
        ) -> torch.Tensor:
        tensor_out = torch.nn.SiLU()(tensor_in)
        tensor_out = self._conv1(tensor_out)
        tensor_out = torch.nn.SiLU()(tensor_out)
        tensor_out = self._conv1(tensor_out)
        return tensor_out + tensor_in

class ResnetStack(BaseModel):
    def __init__(self,
            size_channel_in: int,
            size_channel_out: int,
            size_block: int,
        ):
        super().__init__()
        self._conv = torch.nn.Conv2d(
            in_channels=size_channel_in, 
            out_channels=size_channel_out,
            kernel_size=3,
            stride=1,
            padding="same", 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode="zeros")
        torch.nn.init.xavier_uniform_(self._conv.weight)
        self._pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1)
        self._blocks = torch.nn.ModuleList([])
        for i in range(size_block):
            self._blocks.append(ResnetBlock(size_channel_out))
    def forward(self, 
            tensor_in: torch.Tensor,
        ) -> torch.Tensor:
        tensor_out = self._conv(tensor_in)
        tensor_out = self._pool(tensor_out)
        for i in range(len(self._blocks)):
            tensor_out = self._blocks[i](tensor_out)
        return tensor_out

class ImpalaEncoder(BaseModel):
    def __init__(self,
            size_stacks_channel: List[int],
            size_fc_in: int,
            size_fc_hidden: List[int],
            size_fc_out: int,
            size_block: int,
            use_layer_norm: bool = True,
            dropout_rate: Union[float,None] = None,
        ):
        super().__init__()
        torch._assert(len(size_stacks_channel) > 1, "")
        self._blocks = torch.nn.ModuleList([])
        for i in range(1, len(size_stacks_channel)):
            self._blocks.append(ResnetStack(
                size_channel_in=size_stacks_channel[i-1],
                size_channel_out=size_stacks_channel[i],
                size_block=size_block))
        self._fc = MLPNet(
            size_fc_in, size_fc_hidden, size_fc_out,
            activation=torch.nn.SiLU, 
            spectral_norm_on_hidden=False,
            norm_layer=use_layer_norm)
        if use_layer_norm:
            self._layer_norm1 = torch.nn.LayerNorm(size_fc_in)
            self._layer_norm2 = torch.nn.LayerNorm(size_fc_out)
        else:
            self._layer_norm1 = None
            self._layer_norm2 = None
        if dropout_rate is not None:
            self._dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self._dropout = None
    def forward(self, 
            tensor_in: torch.Tensor,
        ) -> torch.Tensor:
        tensor_out = tensor_in.permute(0,3,1,2).to(torch.float32)/255.0 - 0.5
        for i in range(len(self._blocks)):
            tensor_out = self._blocks[i](tensor_out)
            if self._dropout is not None:
                tensor_out = self._dropout(tensor_out)
        tensor_out = tensor_out.reshape(tensor_out.size(0), -1)
        tensor_out = torch.nn.SiLU()(tensor_out)
        if self._layer_norm1 is not None:
            tensor_out = self._layer_norm1(tensor_out)
        tensor_out = self._fc(tensor_out)
        tensor_out = torch.nn.SiLU()(tensor_out)
        if self._layer_norm2 is not None:
            tensor_out = self._layer_norm2(tensor_out)
        return tensor_out

