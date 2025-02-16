import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial

from core.attention import MemEffAttention, MemEffCrossAttention

class MVAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 4, # WARN: hardcoded!
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*V, C, H, W]
        BV, C, H, W = x.shape
        B = BV // self.num_frames # assert BV % self.num_frames == 0

        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).reshape(BV, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x

class UnetAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        dim_kv: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        #skip_scale: float = 1,
        num_frames: int = 4, # WARN: hardcoded!
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = 1
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffCrossAttention(dim, dim, dim_kv, dim_kv, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
        
        self.post_init()

    def post_init(self):
        nn.init.zeros_(self.attn.proj.weight.data)
        nn.init.zeros_(self.attn.proj.bias.data)

    def forward(self, x, unet_x):
        # x: [B*V, C, H, W]
        BV, C, H, W = x.shape
        #B = BV // self.num_frames # assert BV % self.num_frames == 0

        res = x
        x = self.norm(x)

        x = x.permute(0, 2, 3, 1).reshape(BV, -1, C)
        unet_x = unet_x.permute(0, 2, 3, 1).reshape(BV, H*W, -1)
        x = self.attn(x, unet_x, unet_x)
        x = x.reshape(BV, H, W, C).permute(0, 3, 1, 2)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x
    
class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
        temb_channels: int = 1280,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.nolinearity = F.silu

    def post_init(self):
        nn.init.zeros_(self.time_emb_proj.weight.data)
        nn.init.zeros_(self.time_emb_proj.bias.data)
    
    def forward(self, x, temb=None):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        if temb is not None:
            temb = self.nolinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            x = x + temb
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        unet_out_channels: int,
        unet_out_next_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        unet_attention: bool = False,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()
 
        nets = []
        attns = []
        unet_attns = []
        self.unet_attention = unet_attention

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
            if unet_attention:
                unet_attns.append(UnetAttention(out_channels, unet_out_channels))
            else:
                unet_attns.append(None)

        if unet_attention and downsample:
            self.down_unet_attns = UnetAttention(out_channels, unet_out_next_channels)

        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.unet_attns = nn.ModuleList(unet_attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, unet_xs=None, temb=None):
        xs = []

        for attn, unet_attn, net in zip(self.attns, self.unet_attns, self.nets):
            x = net(x, temb)
            if attn:
                x = attn(x)
            if unet_attn:
                unet_x = unet_xs[0]
                unet_xs = unet_xs[1:]
                x = unet_attn(x, unet_x)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            if unet_attn:
                unet_x = unet_xs[0]
                unet_xs = unet_xs[1:]
                x = self.down_unet_attns(x, unet_x)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x, temb=None):
        x = self.nets[0](x, temb)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x, temb)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs, temb=None):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x, temb)
            if attn:
                x = attn(x)
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x


# it could be asymmetric!
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_unet_channels: Tuple[int, ...] = (320, 320, 320, 640, 1280, 1280),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        down_unet_attention : Tuple[bool, ...] = (False, False, True, True, True, True),
        mid_attention: bool = True,
        #mid_unet_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256),
        #up_unet_channels: Tuple[int, ...] = (1280, 1280, 640, 320, 320),
        up_attention: Tuple[bool, ...] = (True, True, False),
        #up_unet_attention: Tuple[bool, ...] = (True, True, True, True, False),
        #up_last_unet_attention: Tuple[bool, ...] = (False, False, False, True, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]
            unet_cout = down_unet_channels[i]
            unet_next_cout = down_unet_channels[i+1] if i != len(down_channels) - 1 else down_unet_channels[i]
            down_blocks.append(DownBlock(
                cin, cout, unet_cout, unet_next_cout,
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                unet_attention = down_unet_attention[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1],  attention=mid_attention, skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric
            #unet_cout = up_unet_channels[i]

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                #unet_attention = up_unet_attention[i],
                #last_unet_attention = up_last_unet_attention[i],
                skip_scale=skip_scale,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, unet_xss=None, temb=None):
        # x: [B, Cin, H, W]

        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        unet_xss = unet_xss[::-1]
        for block in self.down_blocks:
            if block.unet_attention == True:
                length = len(block.unet_attns) + 1 if block.downsample else len(block.unet_attns)
                unet_xs = unet_xss[:length]
                unet_xss = unet_xss[length:]
                x, xs= block(x, unet_xs, temb)
            else:
                x, xs = block(x, temb)
            xss.extend(xs)
        
        # mid
        # if self.mid_block.unet_attention == True:
        #     unet_xs = unet_xss[0]
        #     unet_xss = unet_xss[1:]
        #     x = self.mid_block(x, unet_xs, temb)
        #else:
        x = self.mid_block(x, temb)

        # up
        for block in self.up_blocks:
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            # if block.unet_attention == True:
            #     length = len(block.unet_attns) + 1 if block.upsample else len(block.unet_attns)
            #     unet_xs = unet_xss[:length]
            #     unet_xss = unet_xss[length:]
            #     x = block(x, xs, unet_xs, temb)
            #else:
            x = block(x, xs, temb)

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']

        return x
