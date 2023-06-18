# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


'''class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)'''


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
#Basic blocks
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k =1, s=1, p='valid' , g=1 , d=1, act= True):
      super().__init__()
      self.conv=nn.Conv2d(c1 ,c2 ,kernel_size=1, stride=s, padding=p, groups=g, dilation =d, bias=False)
      self.bn = nn.BatchNorm2d(c2)
      nn.act=default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

      def forward(self, x):
        return self.act(self.bn(self.conv(x)))
      def forward_plain(self,x):
        return self.act(self.conv(x))
class Bottleneck(nn.Module):
  def __init__(self,c1,c2,skip=True,g=1, e=0.5):
    super().__init__()
    c_=int(c2*e)
    #c_=c2//w
    self.cv1=Conv(c1,c_,1,1)
    self.cv2=Conv(c_,c2,3,1, g=g) #why have we used g param here?

    def forward (self, x):
      return x + self.cv2(self.cv1(x)) if skip else self.cv2(self.cv1(x))

class C3(nn.Module):
  def __init__(self,c1,c2, n=1, shortcut=True, g=1, e=0.5):
    super().__init__()
    c_ = int(c2 * e)  # hidden channels
    self.cv1 = Conv(c1,c_,1,1)
    self.cv2 = Conv(c_, c_,1, 1)
    self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    self.cv3= Conv(2*c_, c2, 1, 1)

    def forward(self,x):
      return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)),1))

class SPPF(nn.Module):
  # equivalent to SPP(k=(5, 9, 13))
  def __init__(self, c1, c2, k=5):
    super().__init__()
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_ * 4, c2, 1, 1)
    self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
      x = self.cv1(x)
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
class Backbone(BackboneBase):
  '''this is a yolov5 backbone'''
  def __init__(self,b=64):
    super().__init__()
    self.layers = nn.Modulelist()
    self.layers+=[ Conv(3, b ,6, 2, 2),
            Conv(b, b*2, 3, 2, 1),
            C3(b*2, b*2, n=2),
            Conv(b*2, b*4, 3, 2, 1),
            C3(b*4, b*4, n=4),
            Conv(b*4, b*8, 3, 2, 1),
            C3(b*8, b*8, n=6),
            Conv(b*8, b*16, 3, 2, 1),
            C3(b*16, b*16, n=2),
            SPPF(b*16, b*16)
        ]
  def forward(self):
    for layer in self.layers:
      x=layer(x)
    return x
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
