# --------------------------------------------------------
# Add BN in convmlp, convmerging, earlyconv embeddings(stride = 1 1 2 2 1).
# Modify Window Attention: BN replaced LN in offset, set k = 3, set WD in entire stages 3 and 4.
# Imagenet Top-1 accuracy: 82.452%
# Food recognition accuracy: 88.91%
# Params: 31.75M
# --------------------------------------------------------
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights
from torch.nn import CrossEntropyLoss
from timm.models.registry import register_model

import math
import einops

class LayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)
        x = self.norm(x)
        x = x.reshape(B, C, H, W)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MlpWithConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels or out_channels
        self.hidden_channels = hidden_channels
        self.fc1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.act = act_layer()
        self.depthwise_conv = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1,
                                        groups=self.hidden_channels)
        self.bn2 = nn.BatchNorm2d(self.hidden_channels)
        self.fc2 = nn.Conv2d(self.hidden_channels, self.out_channels, 1, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.drop(x)

        return x

class PatchAttention(nn.Module):
    def __init__(self, dim, reduction, drop=0.):
        super().__init__()
        self.channels = dim
        self.reduction = reduction

        self.fc1 = nn.Linear(self.channels, self.channels // reduction, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.channels // reduction, 1, bias=False)
        self.norm = nn.LayerNorm(self.channels // reduction)
        self.drop = nn.Dropout(drop)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        :param x:  B N C
        :return:  enhanced feature x
        '''
        x_class = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        B, N, C = x.shape
        score = self.drop(self.norm(self.act(self.fc1(x))))

        score = self.sigmoid(self.drop(self.fc2(score).reshape(B, N))).reshape(B, N, 1)
        x = x * score.expand_as(x)
        x = torch.cat((x_class, x), dim=1)

        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DMSAttention(nn.Module):
    '''
    :param
        dim(int): Number of input channels.
        input_resolution(tuple[int]): Resolution of input features (H, W)
        stride(int): The sample rate of reference_points. (1 or 2)
        postype(str): Type of position encoding. ("conv" or "rel")
        dynamic_factor(int): The transform scale of Dynamic Attention.
        num_heads (int): Number of attention heads.
        k(int): The center points number. (2 or 3)
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, input_resolution, num_heads, stride, dynamic_factor=2, postype="rel", k=None,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.channels = dim
        self.head_channels = self.channels // self.heads
        self.img_size = input_resolution

        self.sample_size = (math.ceil(self.img_size[0] / stride), math.ceil(self.img_size[1] / stride))

        self.factor = dynamic_factor
        self.stride = stride
        self.pos = postype
        self.scale = qk_scale

        self.k = k

        self.offset_mat = None
        self.center_offset = None

        if k is not None:
            if k == 2:
                self.offset_mat = torch.tensor([(0, -1), (-1, 0), (0, 1), (1, 0)])
            elif k == 3:
                self.offset_mat = torch.tensor(
                    [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, 0)])

            self.center_offset = nn.Linear(dim, 2 * k * k, bias=False)
            center_refX = torch.arange(0, self.img_size[0], 1, dtype=torch.float32).view(-1, 1).repeat(1,
                                                                                                       self.img_size[1])
            center_refY = torch.arange(0, self.img_size[1], 1, dtype=torch.float32).view(1, -1).repeat(self.img_size[0],
                                                                                                       1)
            center = torch.stack((center_refX, center_refY), dim=-1)
            self.register_buffer("center", center)

        self.offset = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 3, stride, padding=1, groups=self.channels),
            LayerNorm2D(self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, 2 * self.img_size[0] * self.img_size[1], 1, 1, 0, bias=False)
        )

        self.q_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.k_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.v_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.proj = nn.Conv2d(self.channels, self.channels, 1, 1, 0)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        if self.pos == "conv":
            # depth-wise
            self.posembed = nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels)
        else:  # relative pos embedding
            self.posembed = nn.Parameter(torch.zeros(self.heads, 2 * self.img_size[0] - 1, 2 * self.img_size[1] - 1))

            ref_pos = (torch.arange(0, self.img_size[0], self.stride, dtype=torch.float32).view(1, -1)
                       - torch.arange(0, self.img_size[1], 1, dtype=torch.float32).view(-1, 1))

            # ref_pos_x: 列偏移   ref_pos_y: 行偏移
            ref_pos_x = ref_pos.repeat(self.img_size[0], self.sample_size[0])
            ref_pos_y = ref_pos.repeat_interleave(self.img_size[1], dim=0).repeat_interleave(self.sample_size[1], dim=1)

            ref_pos_point = torch.stack((ref_pos_y, ref_pos_x), dim=-1)
            ref_pos_point[..., 0] = ref_pos_point[..., 0].mul(1.0).div(self.img_size[0] - 1)
            ref_pos_point[..., 1] = ref_pos_point[..., 1].mul(1.0).div(self.img_size[1] - 1)
            self.register_buffer("ref_pos_point", ref_pos_point)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def reference_points(self, batch, H, W, rH, rW, stride, dtype=torch.float32, device=None):
        refX = torch.arange(0, H, stride, dtype=dtype, device=device).view(-1, 1).repeat(1, rW)
        # print(refX.shape)
        refY = torch.arange(0, W, stride, dtype=dtype, device=device).view(1, -1).repeat(rH, 1)
        # print(refY)

        center = None
        if self.offset_mat is not None:
            center = self.center
            center = center.reshape(H * W, 2).unsqueeze(dim=0)
            center = center.expand(self.k * self.k, H * W, 2).permute(1, 0, 2)

            offset_mat = self.offset_mat.unsqueeze(dim=0).expand(H * W, self.k * self.k, 2)
            center = center + offset_mat
            center[..., 0] = torch.clip(center[..., 0], 0, H).div(H - 1).mul(2).sub(1)
            center[..., 1] = torch.clip(center[..., 1], 0, W).div(W - 1).mul(2).sub(1)
            center = center.unsqueeze(dim=0).expand(batch, H * W, self.k * self.k, 2).reshape(batch * H * W, self.k,
                                                                                              self.k, 2)
            # 按照batch->像素的顺序排列

        grid = torch.stack((refX, refY), dim=-1)
        grid[..., 0] = grid[..., 0].div(H - 1).mul(2).sub(1)
        grid[..., 1] = grid[..., 1].div(W - 1).mul(2).sub(1)
        grid = grid.unsqueeze(dim=0).expand(batch * H * W, rH, rW, 2)

        return grid, center

    def forward(self, x):
        # x= x.cuda()
        B, C, H, W = x.shape

        rH = math.ceil(H / self.stride)
        rW = math.ceil(W / self.stride)

        # ref_points:（B*H*W, rH, rW, 2） 300000 /6122   center_points: (B*H*W, k, k, 2)     14112  k=9
        ref_points, center_points = self.reference_points(B, H, W, rH, rW, self.stride)
        # print(ref_points[0])

        # query: (B, C, H, W)   301056
        query = self.q_conv(x)
        # offset:(B, 2*H*W, rH, rW)  300000/6122
        offset = self.offset(query)
        # offset:(B, 2, H*W, rH*rW)  300000/6122
        offset = einops.rearrange(offset, 'b (g h w) rh rw -> b g (h w) (rh rw)', g=2, h=H, w=W, rh=rH, rw=rW)

        # norm the offset to (-1,1)
        ranges = torch.tensor([1.0 / H, 1.0 / W]).view([1, 2, 1, 1])
        offset = self.tanh(offset).mul(ranges)

        if self.factor > 1:
            offset = offset.mul(self.factor)

        # offset:（B*H*W, rH, rW, 2）   第一维为行偏移，第二维为列偏移。
        offset = offset.permute(0, 2, 3, 1).reshape(B * H * W, rH, rW, 2)  # 排列顺序:第一个batch的HW个像素，然后第二个batch的HW个像素，...
        dpos = offset + ref_points

        # expand_x（B*H*W, C, H, W)
        expand_x = x.reshape(B, C, H, W).expand(H * W, B, C, H, W)  # 236M/4M
        expand_x = einops.rearrange(expand_x, '(h w) b c H W->(b h w) c H W', h=H, w=W, b=B)
        # grid_sample: (B*H*W, C, rH, rW)    59M/1.2M
        grid_sample = F.grid_sample(
            input=expand_x,
            grid=dpos[..., (1, 0)],
            mode="bilinear",
            align_corners=True
        )

        sample = rH * rW

        # k,v: (B*head, H*W, head_channels, rH*rW)   #59M/1.2M*2=2.4M
        k = self.k_conv(grid_sample).reshape(B, H * W, C, rH * rW).permute(0, 2, 1, 3).reshape(B * self.heads,
                                                                                               self.head_channels,
                                                                                               H * W, rH * rW)
        v = self.v_conv(grid_sample).reshape(B, H * W, C, rH * rW).permute(0, 2, 1, 3).reshape(B * self.heads,
                                                                                               self.head_channels,
                                                                                               H * W, rH * rW)

        if self.offset_mat is not None:
            sample = sample + self.k * self.k
            # [batch, H*W, k, k, 2]
            center_offset = self.center_offset(query.permute(0, 2, 3, 1).reshape(B, H * W, C))
            center_offset = einops.rearrange(center_offset, 'b N (h w s) -> (b N) h w s', N=H * W, h=self.k, w=self.k,
                                             s=2)

            center_offset = self.tanh(center_offset).mul(ranges.view(1, 1, 1, 2))
            center_grid = center_offset + center_points

            # (B*H*W, C, k, k)
            center_sample = F.grid_sample(
                input=expand_x,
                grid=center_grid[..., (1, 0)],
                mode="bilinear",
                align_corners=True
            )
            # print(center_sample.shape)
            center_k = self.k_conv(center_sample).reshape(B, H * W, C, self.k * self.k). \
                permute(0, 2, 1, 3).reshape(B * self.heads, self.head_channels, H * W, self.k * self.k)
            center_v = self.v_conv(center_sample).reshape(B, H * W, C, self.k * self.k). \
                permute(0, 2, 1, 3).reshape(B * self.heads, self.head_channels, H * W, self.k * self.k)

            k = torch.cat((k, center_k), dim=-1)
            v = torch.cat((v, center_v), dim=-1)

        query = query.reshape(B * self.heads, self.head_channels, H * W).unsqueeze(dim=-1).permute(0, 2, 3, 1)

        if self.scale:
            query = query * self.scale
        attn = query @ (k.permute(0, 2, 1, 3))

        dwpos = None
        if self.pos == "conv":
            dwpos = self.posembed(query.reshape(B, C, H, W))
        else:
            relative_dynamic_point = self.ref_pos_point
            relative_dynamic_position = relative_dynamic_point.reshape(H * W, rH, rW, 2)[None, ...].expand(B, H * W, rH,
                                                                                                           rW, 2)
            relative_dynamic_position = relative_dynamic_position.reshape(B * H * W, rH, rW,
                                                                          2)  # 按照batch的像素排序，第一个batch的HW个像素,...
            relative_dynamic_position -= offset

            rel_pos_tab = self.posembed[None, ...].expand(B * H * W, self.heads, 2 * H - 1, 2 * W - 1)
            d_rel_pos = F.grid_sample(input=rel_pos_tab,
                                      grid=relative_dynamic_position[..., (1, 0)],
                                      mode="bilinear",
                                      align_corners=True
                                      )
            # （B*H*W，head，rH，rW）
            d_rel_pos = d_rel_pos.reshape(B, H * W, self.heads, rH * rW).permute(0, 2, 1, 3).unsqueeze(dim=3).reshape(
                B * self.heads, H * W, 1, rH * rW)

            # (H,W,2)
            center = self.center
            center = center.reshape(H * W, 2).unsqueeze(dim=0)
            # (H*W, k*k, 2)
            center = center.expand(self.k * self.k, H * W, 2).permute(1, 0, 2)
            center = center[None, ...].expand(B, H * W, self.k * self.k, 2).reshape(B * H * W, self.k, self.k, 2)
            center_rel_pos = center_offset - center

            d_center_rel_pos = F.grid_sample(input=rel_pos_tab,
                                             grid=center_rel_pos[..., (1, 0)],
                                             mode="bilinear",
                                             align_corners=True
                                             )
            d_center_rel_pos = d_center_rel_pos.reshape(B, H * W, self.heads, self.k * self.k).permute(0, 2, 1, 3). \
                unsqueeze(dim=3).reshape(B * self.heads, H * W, 1, self.k * self.k)

            d_rel_pos = torch.cat((d_rel_pos, d_center_rel_pos), dim=-1)
            attn = attn + d_rel_pos

        # attn: (batch*head, H*W, 1, rH*rW)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # x: (batch*head, H*W, 1, head_Channels)
        x = attn @ (v.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2).squeeze(dim=-1)
        x = x.reshape(B, C, H, W)

        if self.pos == "conv":
            x = x + dwpos

        # x: (batch, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, dpos

class Window_DMSAttention(nn.Module):
    '''
    :param
        dim(int): Number of input channels.
        input_resolution(tuple[int]): Resolution of input features (H, W)
        stride(int): The sample rate of reference_points. (1 or 2)
        postype(str): Type of position encoding. ("conv" or "rel")
        dynamic_factor(int): The transform scale of Dynamic Attention.
        num_heads (int): Number of attention heads.
        k(int): The center points number. (2 or 3)
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, input_resolution, num_heads, stride, dynamic_factor=2, window_size=7,
                 postype="rel", k=None, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.channels = dim
        self.head_channels = self.channels // self.heads
        self.head_per_group = 4
        self.head_groups = num_heads // self.head_per_group
        self.groups_channel = self.channels // self.head_groups
        self.img_size = input_resolution
        self.window_size = window_size
        self.window_num = (int(input_resolution[0] / window_size), int(input_resolution[1] / window_size))

        self.sample_size = (math.ceil(self.img_size[0] / stride), math.ceil(self.img_size[1] / stride))

        self.factor = dynamic_factor
        self.stride = stride
        self.pos = postype
        self.scale = qk_scale or dim ** -0.5

        self.k = k

        self.offset_mats = None
        self.center_offset = None

        if k is not None:
            if k == 2:
                self.offset_mats = torch.tensor([(0, -1), (-1, 0), (0, 1), (1, 0)])
            elif k == 3:
                self.offset_mats = torch.tensor(
                    [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, 0)])
            #compute the final offset.
            self.register_buffer("offset_mat", self.offset_mats)

            self.center_offset = nn.Sequential(
                nn.Conv2d(self.groups_channel, self.groups_channel, 3,1,1, groups=self.groups_channel),
                nn.BatchNorm2d(self.groups_channel),
                nn.GELU(),
                nn.Conv2d(self.groups_channel, 2*k*k, 1,1,0)
            )
            #compute the relative position embedding
            center_refX = torch.arange(0, self.img_size[0], 1).view(-1, 1).repeat(1, self.img_size[1])
            center_refY = torch.arange(0, self.img_size[1], 1).view(1, -1).repeat(self.img_size[0], 1)
            center = torch.stack((center_refX, center_refY), dim=-1)
            self.register_buffer("center", center)

        self.offset = nn.Sequential(
            nn.Conv2d(self.groups_channel, self.groups_channel, window_size, window_size, padding=0,
                      groups=self.groups_channel, bias=False),
            nn.BatchNorm2d(self.groups_channel),
            nn.GELU(),
            nn.Conv2d(self.groups_channel, 2*self.sample_size[0]*self.sample_size[1], 1, 1, 0,
                      bias=False)
        )

        self.q_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.k_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.v_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        if k is not None:
            self.proj = nn.Conv2d(2 * self.channels, self.channels, 1, 1, 0)
        else:
            self.proj = nn.Conv2d(self.channels, self.channels, 1, 1, 0)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        if self.pos == "conv":
            # depth-wise
            self.posembed = nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels)
        else:  # relative pos embedding
            self.posembed = nn.Parameter(torch.zeros(self.heads, 2 * self.img_size[0] - 1, 2 * self.img_size[1] - 1))
            trunc_normal_(self.posembed, std=0.01)

            ref_pos = (torch.arange(0, self.img_size[0], self.stride).view(1, -1)
                       - torch.arange(0, self.img_size[1], 1).view(-1, 1))

            # ref_pos_x: 列偏移   ref_pos_y: 行偏移
            ref_pos_x = ref_pos.repeat(self.img_size[0], self.sample_size[0])
            ref_pos_y = ref_pos.repeat_interleave(self.img_size[1], dim=0).repeat_interleave(self.sample_size[1], dim=1)

            ref_pos_point = torch.stack((ref_pos_y, ref_pos_x), dim=-1)
            self.register_buffer("ref_pos_point", ref_pos_point)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def reference_points(self, batch, H, W, rH, rW, stride, dtype=torch.float32, device=None):
        refX = torch.arange(0, H, stride, dtype=dtype, device=device).view(-1, 1).repeat(1, rW)
        # print(refX.shape)
        refY = torch.arange(0, W, stride, dtype=dtype, device=device).view(1, -1).repeat(rH, 1)
        # print(refY)

        center = None
        if self.offset_mat is not None:
            center = self.center
            center = center.type(dtype)
            center = center.reshape(H * W, 2).unsqueeze(dim=0)
            center = center.expand(self.k * self.k, H * W, 2).permute(1, 0, 2)

            offset_mat = self.offset_mat.unsqueeze(dim=0).expand(H * W, self.k * self.k, 2)
            center = center + offset_mat
            center[..., 0] = torch.clip(center[..., 0], 0, H-1).div(H - 1).mul(2).sub(1)
            center[..., 1] = torch.clip(center[..., 1], 0, W-1).div(W - 1).mul(2).sub(1)
            center = center.unsqueeze(dim=0).expand(batch * self.head_groups, H * W, self.k * self.k, 2)


        grid = torch.stack((refX, refY), dim=-1)
        grid[..., 0] = grid[..., 0].div(H - 1).mul(2).sub(1)
        grid[..., 1] = grid[..., 1].div(W - 1).mul(2).sub(1)
        grid = grid.unsqueeze(dim=0).expand(batch * self.head_groups * self.window_num[0] * self.window_num[1], rH, rW, 2)

        return grid, center

    def forward(self, x):
        # x= x.cuda()
        B, C, H, W = x.shape
        Wh, Ww = self.window_num[0], self.window_num[1]
        Ws = self.window_size
        dtype = x.dtype
        device = x.device

        rH = math.ceil(H / self.stride)
        rW = math.ceil(W / self.stride)

        h = self.heads
        h_c = self.head_channels

        hg = self.head_groups
        gc = self.groups_channel
        hpg = self.head_per_group

        # ref_points:（B*hg*Wh*Ww, rH, rW, 2） 9408   center_points: (B*hg, H*W, k*k, 2)     37632  k=9
        ref_points, center_points = self.reference_points(B, H, W, rH, rW, self.stride, dtype=dtype, device=device)

        # (B*hg, Wh*Ww, rH*rW, 2)
        ref_points = einops.rearrange(ref_points, '(b h Wh Ww) rh rw g -> (b h) (Wh Ww) (rh rw) g',
                                      b=B, h=hg, Wh=Wh, Ww=Ww, rh=rH, rw=rW)
        # print(ref_points.shape)

        # query: (B, C, H, W)   0.15M/75264/38416
        query = self.q_conv(x)

        # offset:(B*hg, 2*rh*rw, Wh, Ww)   460000/9408
        offset = self.offset(query.reshape(B*hg, gc, H, W))
        # offset:(B, 2, hg, Wh*Ww, rH*rW)  460000/9408
        offset = offset.reshape(B*hg, 2, rH*rW, Wh*Ww).permute(0, 3, 2, 1)

        #offset = einops.rearrange(offset, 'b (g h Wh Ww) rh rw -> b g h (Wh Ww) (rh rw)', g=2, h=hg, Wh=Wh, Ww=Ww,rh=rH, rw=rW)
        #print(offset.shape)
        # norm the offset to (-1,1)
        ranges = torch.tensor([1.0 / H, 1.0 / W], device=device).view([1,1,1,2])
        offset = self.tanh(offset).mul(ranges)

        if self.factor > 1:
            offset = offset.mul(self.factor)
        # offset:(B, hg, Wh*Ww, rH*rW, 2) ->（B*hg, Wh*Ww, rH*rW, 2）   第一维为行偏移，第二维为列偏移。
        #offset = offset.permute(0, 2, 3, 4, 1).reshape(B * hg, Wh * Ww, rH * rW, 2)  # 排列顺序:第一个batch的HW个像素，然后第二个batch的HW个像素，...
        sample_grid = offset + ref_points
        # print(sample_grid.shape)

        # value（B, h, h_c, H, W)   #0.15M
        value = x.reshape(B, hg, gc, H, W).view(B * hg, gc, H, W)
        # grid_sample: (B*hg, gc, Wh*Ww, rH*rW)    0.15M
        grid_sample = F.grid_sample(
            input=value,
            grid=sample_grid[..., (1, 0)],
            mode="bilinear",
            align_corners=True
        )
        # print(grid_sample.shape)

        grid_sample = grid_sample.reshape(B, C, Wh * Wh, rH * rW)

        # k,v: (B*h, h_c, Wh*Ww, rH*rW)   #7M/0.15M*2=0.3M  -> (B*h, Wh*Ww, h_c, rH*rW)
        k = self.k_conv(grid_sample).reshape(B * h, h_c, Wh * Wh, rH * rH).permute(0, 2, 1, 3)
        v = self.v_conv(grid_sample).reshape(B * h, h_c, Wh * Wh, rH * rH).permute(0, 2, 1, 3)
        # print(k.shape)

        # (B*h, H, W, h_c)
        window_query = query.reshape(B * h, h_c, H, W).permute(0, 2, 3, 1)
        # B*h*Wh*Ww, W_s, W_s, C
        window_query = window_partition(window_query, self.window_size).reshape(B * h, Wh * Ww,
                                                                                self.window_size * self.window_size,
                                                                                h_c)
        # print(window_query.shape)

        if self.scale:
            window_query = window_query * self.scale
        attn = window_query @ k

        center_output = None
        if self.offset_mat is not None:
            ks = self.k
            #(batch*hg, 2*k*k, H, W)
            center_offset = self.center_offset(query.reshape(B*hg, gc, H, W)).reshape(B*hg, 2, ks*ks, H*W).permute(0, 3, 2, 1)
            #print(center_offset.shape)
            # [Batch, H, W, 2*hg*k*k] -> [Batch, H*W, h, k, k, 2] -> [B*hg, H * W, k * k, 2]
            #center_offset = center_offset.reshape(B, H * W, 2, hg, ks * ks).permute(0, 3, 1, 4, 2).reshape(B * hg, H * W, ks * ks, 2)

            center_offset = self.tanh(center_offset).mul(
                torch.tensor([1.0 / H, 1.0 / W], device=device).view(1, 1, 1, 2))
            #(B*hg, H*W, k*k ,2)
            center_grid = center_offset + center_points

            # (B*hg, gc, H*W, k*k)    1.3M(k=3)/0.6M(k=2)
            center_sample = F.grid_sample(
                input=value,
                grid=center_grid[..., (1, 0)],
                mode="bilinear",
                align_corners=True
            )
            # print(center_sample.shape)   [B, C, H*W, k*k]
            center_sample = einops.rearrange(center_sample, '(b h) hc A B->b (h hc) A B', b=B, h=hg, hc=gc)

            center_k = self.k_conv(center_sample).reshape(B * h, h_c, H * W, ks * ks)
            center_v = self.v_conv(center_sample).reshape(B * h, h_c, H * W, ks * ks)

            # (B*h, H*W, 1, h_c)
            center_q = query.reshape(B * h, h_c, H * W).unsqueeze(dim=2).permute(0, 3, 2, 1)
            if self.scale:
                center_q = center_q * self.scale
            # (B*h, H*W, 1, k*k)   18816
            center_attn = center_q @ (center_k.permute(0, 2, 1, 3))

            if self.pos == "rel":
                center = self.center
                center = center.type(dtype)
                center = center.reshape(H * W, 2).unsqueeze(dim=0)
                # (H*W, k*k, 2)
                center = center.expand(self.k * self.k, H * W, 2).permute(1, 0, 2)
                center = center[None, ...].expand(B * hg, H * W, self.k * self.k, 2)
                center_rel_pos = center_offset - center

                # 18816
                rel_pos_tab = self.posembed[None, ...].expand(B, h, 2 * H - 1, 2 * W - 1).reshape(B * hg, hpg,
                                                                                                  2 * H - 1, 2 * W - 1)
                # (B*hg, hpg, H*W, k*k)     42336
                d_center_rel_pos = F.grid_sample(input=rel_pos_tab,
                                                 grid=center_rel_pos[..., (1, 0)],
                                                 mode="bilinear",
                                                 align_corners=True
                                                 )
                # (B*h, 1, HW, k*k)
                d_center_rel_pos = d_center_rel_pos.reshape(B * h, 1, H * W, self.k * self.k).permute(0, 2, 1, 3)
                center_attn = center_attn + d_center_rel_pos

            center_attn = self.softmax(center_attn)

            center_attn = self.attn_drop(center_attn)
            # 0.15M   (B/h, H*W, 1, h_c)
            center_output = center_attn @ (center_v.permute(0, 2, 3, 1))
            center_output = center_output.permute(0, 3, 1, 2).squeeze(dim=-1).reshape(B, C, H, W)

        dwpos = None
        if self.pos == "conv":
            dwpos = self.posembed(query)
        else:
            # (H*W, rH*rW, 2)
            relative_dynamic_point = self.ref_pos_point
            relative_dynamic_point = relative_dynamic_point.type(dtype)
            relative_dynamic_point[..., 0] = relative_dynamic_point[..., 0].mul(1.0).div(self.img_size[0] - 1)
            relative_dynamic_point[..., 1] = relative_dynamic_point[..., 1].mul(1.0).div(self.img_size[1] - 1)
            relative_dynamic_point = relative_dynamic_point[None, ...].expand(B * hg, H * W, rH * rW, 2).clone()
            

            # (B*hg, Wh*Ww, rH*rW, 2）->(B*hg, H*W, rH*rW, 2)  #1.2M
            offset = offset.reshape(B * hg, Wh, Ww, rH * rW, 2).repeat_interleave(self.window_size,
                                                                                  dim=1).repeat_interleave(
                self.window_size, dim=2)
            offset = offset.reshape(B * hg, H * W, rH * rW, 2)
            relative_dynamic_point -= offset

            rel_pos_tab = self.posembed[None, ...].expand(B, h, 2 * H - 1, 2 * W - 1).reshape(B * hg, hpg, 2 * H - 1,
                                                                                              2 * W - 1)  #
            # （B*hg，hpg，H*W，rH*rW） 1.2M
            d_rel_pos = F.grid_sample(input=rel_pos_tab,
                                      grid=relative_dynamic_point[..., (1, 0)],
                                      mode="bilinear",
                                      align_corners=True
                                      )
            # （B*h，H*W，rH*rW） 1.2M
            d_rel_pos = d_rel_pos.reshape(B, h, H, W, rH * rW).reshape(B * h, H, W, rH * rW)
            d_rel_pos = d_rel_pos.view(B * h, H // self.window_size, self.window_size, W // self.window_size,
                                       self.window_size, rH * rW)
            # (B*h, Wh*Ww, Ws*Ws, rH*rW)
            d_rel_pos = d_rel_pos.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B * h, Wh * Ww,
                                                                                 self.window_size * self.window_size,
                                                                                 rH * rW)

            # attn = attn + d_rel_pos
            attn = attn + d_rel_pos

        # attn: (batch*head, Wh*Ww,Ws*Ws,rH*rW)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # x: (batch*head, Wh*Ww, Ws*Ws, head_Channels)
        x = attn @ (v.permute(0, 1, 3, 2))
        x = x.reshape(B, h, Wh * Ww, self.window_size * self.window_size, h_c).permute(0, 2, 3, 1, 4).reshape(
            B * Wh * Ww, -1, C)

        x = window_reverse(x, self.window_size, H, W).permute(0, 3, 1, 2)

        if self.pos == "conv":
            x = x + dwpos

        if self.offset_mat is not None:
            x = torch.cat((x, center_output), dim=1)
        # x: (batch, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, sample_grid

class ConvPatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, reduction=True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = self.dim * 4
        self.output_dim = self.dim * 2
        self.norm1 = norm_layer(self.hidden_dim)
        self.norm2 = norm_layer(self.output_dim)

        if reduction:
            self.sample = nn.Sequential(
                nn.Conv2d(self.dim, self.hidden_dim, 3, 2, 1, groups=self.dim),
                nn.GELU(),
                self.norm1,
                nn.Conv2d(self.hidden_dim, self.dim * 2, 1, 1, 0),
                nn.GELU(),
                self.norm2
            )
        else:
            self.sample = nn.Sequential(
                nn.Conv2d(self.dim, self.hidden_dim, 3, 2, 1),
                nn.GELU(),
                self.norm1,
                nn.Conv2d(self.hidden_dim, self.dim * 2, 1, 1, 0),
                nn.GELU(),
                self.norm2
            )

    def forward(self, x):
        x = self.sample(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        x = x.reshape(B, H / 2, W / 2, 4 * C)
        x = x.permute(0, 3, 1, 2)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # B C H W
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchEmbedWithConv(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        channel_list = [48, 96, 192, 384]
        stride = None

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        if patch_size[0] == 4:
            stride = [1, 1, 2, 2, 1]
        if patch_size[0] == 8:
            stride = [1, 2, 2, 2, 1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim


        #self.proj = nn.Conv2d(in_chans, embed_dim, 7, 4, 2)

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, channel_list[0], kernel_size=3, stride=stride[0], padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.GELU(),
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, stride=stride[1], padding=1),
            nn.BatchNorm2d(channel_list[1]),
            nn.GELU(),
            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, stride=stride[2], padding=1),
            nn.BatchNorm2d(channel_list[2]),
            nn.GELU(),
            nn.Conv2d(channel_list[2], channel_list[3], kernel_size=3, stride=stride[3], padding=1),
            nn.BatchNorm2d(channel_list[3]),
            nn.GELU(),
            nn.Conv2d(channel_list[3], embed_dim, kernel_size=1, stride=stride[4], padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        if norm_layer is not None:
            self.norm = LayerNorm2D(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class DMSTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., stride=2, dynamic_factor=2,
                 Attention_type="D", postype="rel", window_size=7, shift_size=0, conv_mlp = True,
                 qkv_bias=True, qk_scale=None, k=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNorm2D):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.stride = stride
        self.dynamic_factor = dynamic_factor
        self.window_size = window_size
        self.shift_size = shift_size
        self.Attention_type = Attention_type

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        assert postype in ["conv", "rel"], "position embeddings not found!"

        self.norm1 = norm_layer(dim)

        if Attention_type == "D":
            self.attn = DMSAttention(
                dim, input_resolution, num_heads=num_heads, stride=stride, k=k, dynamic_factor=dynamic_factor,
                postype=postype, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif Attention_type == "WD":
            self.attn = Window_DMSAttention(dim, input_resolution, num_heads=num_heads, stride=stride, k=k,
                                            window_size=self.window_size, dynamic_factor=dynamic_factor,
                                            postype=postype,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif Attention_type == "W":
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution
                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if conv_mlp:
            self.mlp = MlpWithConv(in_channels=dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.Attention_type == "D":
            # Dynamic attention
            x, _ = self.attn(x)  # B, C, H, W

        elif self.Attention_type == "WD":
            x, _ = self.attn(x)

        elif self.Attention_type == "W":
            # B, H, W, C
            x = x.permute(0, 2, 3, 1)
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            # B, C, H, W
            x = x.permute(0, 3, 1, 2)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DMSTransformerstage(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Num
        ber of attention heads.
        window_size (int): Local window size.
        Attention_type(List[char]): The Attention type of each Layer. ("W" or "D")
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, Attention_type, postype="rel", mlp_ratio=4., conv_mlp = True,
                 window_size=7, stride=2, dynamic_factor=2, qkv_bias=True, qk_scale=None, center_k=2, drop=0.,
                 attn_drop=0.,drop_path=0., norm_layer=LayerNorm2D, downsample=ConvPatchMerging, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DMSTransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                conv_mlp = conv_mlp,
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                Attention_type=Attention_type[i],
                                stride=stride,
                                dynamic_factor=dynamic_factor,
                                postype=postype,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                k=center_k,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class DLETransformer(nn.Module):
    '''
    :param
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        Attention_type(tuple(string)): the Attention type of each stage.
        dynamic_factor(int): the expand factor of Dynamic Attention. Default: 2
        dynamic_stride(int): the stride of downsampling of Dynamic Attention. Default: 2 (1 means don't downsample)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    '''

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7,
                 Attention_type=[['W','W'],["W","W"],["W","W","W","W","W","W","W","W","W","W","W","W","WD","WD","WD","WD","WD","WD"],["WD","WD"]],
                 dynamic_factor=[None, 2, 2, 2], dynamic_stride=[None, 4, 2, 2], center_k=3, postype="rel",
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., conv_mlp = True,
                 drop_path_rate=0.1, conv_pe=True, conv_pm=True, norm_layer=LayerNorm2D, ape=False,
                 dwpe=True, patch_norm=True, use_checkpoint=False, weight_init='jax', **kwargs):

        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.dwpe = dwpe
        self.conv_pm = conv_pm
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        if conv_pe:
            self.patch_embed = PatchEmbedWithConv(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # CPVT卷积位置编码
        if self.dwpe:
            self.absolute_pos_embed = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1, bias = True, groups=self.embed_dim)

        if self.conv_pm:
            self.PatchMerge = ConvPatchMerging
        else:
            self.PatchMerge = PatchMerging

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        layers = []
        for i_layer in range(self.num_layers):
            layers += [DMSTransformerstage(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                Attention_type=Attention_type[i_layer],
                postype=postype,
                mlp_ratio=self.mlp_ratio,
                conv_mlp=conv_mlp,
                window_size=window_size,
                stride=dynamic_stride[i_layer],
                dynamic_factor=dynamic_factor[i_layer],
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                center_k=center_k,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=self.PatchMerge if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            ]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        if weight_init.startswith('jax'):
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            self.apply(_init_vit_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table','posembed',"bn"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        if self.ape:
            x = x.shape(B, C, H * W).permute(0, 2, 1)
            x = x + self.absolute_pos_embed
            x = x.permute(0, 1, 2).reshape(B, C, H, W)

        if self.dwpe:
            pos = self.absolute_pos_embed(x)
            x = pos + x

        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B C H W
        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



'''
def swin_small_patch4_window7_224(**kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return SwinTransformer(**model_kwargs)

def swin_base_patch4_window7_224(pretrained = "", **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model = SwinTransformer(**model_kwargs)
    state = torch.load(pretrained, map_location='cpu')
    filtered_dict = {k: v for k, v in state['model'].items() if k != "head.weight" and k != "head.bias"}
    model.load_state_dict(filtered_dict, strict=False)
    return model
'''
'''
model = swin_base_patch4_window7_224(pretrained="E:/Code/Swin-Transformer-main/swin_base_patch4_window7_224.pth",
                                     num_classes = 101,
                                     drop_path_rate = 0.1,
                                     drop_rate = 0.0
                                     )

inputs = torch.randn([3,3,224,224])
print(model(inputs).shape)
input()
'''
torch.set_printoptions(profile="full")
model = DLETransformer(
)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))

inputs = torch.randn([2,3,224,224])
print(model(inputs).shape)

