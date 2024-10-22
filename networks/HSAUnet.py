import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
# from timm.models.layers import DropPath, trunc_normal_
from networks.segformer import *
import math
import networks.gap as gap


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(
        keep_prob) * random_tensor  # 若要保持期望和不使用dropout时一致，就要除以p。   参考网址：https://www.cnblogs.com/dan-baishucaizi/p/14703263.html
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(self, tensor, mean=0, std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    '''
    Mix FFN with residual
    '''

    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp=""):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)

        attn = self.attn(norm_1, norm_2)
        # attn = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(attn)

        # residual1 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x1)
        # residual2 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


class DualTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, token_mlp=""):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # embed_dims=[64, 128, 256, 512]
        # num_heads=[1, 2, 4, 8]
        # mlp_ratios=[4, 4, 4, 4]
        # sr_ratios=[8, 4, 2, 1]
        # block = nn.ModuleList([Block(
        #     dim=embed_dims[i], mask=True if (j % 2 == 1 and i < num_stages - 1) else False, num_heads=num_heads[i],
        #     mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[i], linear=linear)
        #     for j in range(depths[i])])
        self.hybrid_scaled_attention = Attention(dim, False, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 attn_drop=attn_drop, proj_drop=0,
                                                 sr_ratio=sr_ratio)
        # 转换成通道注意力机制，
        # 因为发现根据重要性程度划分的话，有可能将肾部作为不重要区域，导致对肾部识别率差
        # self.self_guide_attention = Attention(dim, True, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=0, sr_ratio=sr_ratio)
        self.channel_attn = ChannelAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(dim, int(dim * 4))
            self.mlp2 = MixFFN(dim, int(dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(dim, int(dim * 4))
            self.mlp2 = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp1 = MLP_FFN(dim, int(dim * 4))
            self.mlp2 = MLP_FFN(dim, int(dim * 4))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        # norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn, mask = self.hybrid_scaled_attention(norm1, H, W, None)
        # attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        # mlp1 = self.mlp1(norm2, H, W)
        mlp1 = self.mlp1(norm2)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        # self_guide_attn, mask = self.self_guide_attention(norm3, H, W, mask)
        self_guide_attn = self.channel_attn(norm3)

        add3 = add2 + self_guide_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4)

        mx = add3 + mlp2
        return mx


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wh, Ww)
        return x


class PatchEmbed(nn.Module):
    '''
    Embedding layer
    '''

    def __init__(self, stride1=2, stride2=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = stride1
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # stride1 = [patch_size // 1]
        # stride2 = [patch_size // 2]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))
        x = self.proj1(x)  # B C Ws Wh Ww
        x = self.proj2(x)  # B C Ws Wh Ww
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2).contiguous()
        # if self.norm is not None:
        #     Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        #     x = x.flatten(2).transpose(1, 2).contiguous()
        #     x = self.norm(x)
        #     x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x, H, W


# Encoder
class Encoders(nn.Module):
    def __init__(self, image_size, in_dim, heads, sr_ratios, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()

        self.patch_embed1 = PatchEmbed(stride1=2, stride2=2, in_chans=3, embed_dim=in_dim[0])
        self.patch_embed2 = PatchEmbed(stride1=2, stride2=1, in_chans=in_dim[0], embed_dim=in_dim[1])
        self.patch_embed3 = PatchEmbed(stride1=2, stride2=1, in_chans=in_dim[1], embed_dim=in_dim[2])

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], heads[0], sr_ratio=sr_ratios[0]) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], heads[1], sr_ratio=sr_ratios[1]) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        # a global attention bottleneck
        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], heads[2], sr_ratio=sr_ratios[2]) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

        # 2 is num_class
        out_planes = 2 * 8
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(512, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.direc_reencode = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1),
            # nn.BatchNorm2d(out_planes),
            # nn.ReLU(True)
        )
        self.gap = gap.GlobalAvgPool2D()
        # self.sde_module = SDE_module(512, 512, out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # #### directional Prior ####
        # directional_c5 = self.channel_mapping(x)
        # mapped_c5 = F.interpolate(directional_c5, scale_factor=32, mode='bilinear', align_corners=True)
        # mapped_c5 = self.direc_reencode(mapped_c5)
        #
        # d_prior = self.gap(mapped_c5)
        # x = self.sde_module(x, d_prior)

        outs.append(x)

        return outs #, mapped_c5


def local_conv(dim):
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)


def window_partition(x, window_size, H, W):
    B, num_heads, N, C = x.shape
    x = x.contiguous().view(B * num_heads, N, C).contiguous().view(B * num_heads, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C). \
        view(-1, window_size * window_size, C)
    return windows  # (B*numheads*num_windows, window_size, window_size, C)


def window_reverse(windows, window_size, H, W, head):
    Bhead = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(Bhead, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(Bhead, H, W, -1).view(Bhead // head, head, H, W, -1) \
        .contiguous().permute(0, 2, 3, 1, 4).contiguous().view(Bhead // head, H, W, -1).view(Bhead // head, H * W, -1)
    return x  # (B, H, W, C)


class Attention(nn.Module):
    def __init__(self, dim, mask, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            if mask:
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                if self.sr_ratio == 8:
                    f1, f2, f3 = 14 * 14, 56, 28
                elif self.sr_ratio == 4:
                    f1, f2, f3 = 49, 14, 7
                elif self.sr_ratio == 2:
                    f1, f2, f3 = 2, 1, None
                self.f1 = nn.Linear(f1, 1)
                self.f2 = nn.Linear(f2, 1)
                if f3 is not None:
                    self.f3 = nn.Linear(f3, 1)
            else:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02, mean=0)
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask):
        B, N, C = x.shape
        # 1. X are projected into Q, K, V
        # lepe_liner: B, N, C -> B, N, C.
        ## transpose: B, N, C -> B, C, N
        ## view: B, C, N -> B, C, H, W
        # lepe_conv: B, C, H, W -> ???
        ## view: B, C, H, W -> B, C, N
        ## transpose: B, C, N -> B, N, C
        lepe = self.lepe_conv(
            self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)).view(B, C, -1).transpose(-1, -2)
        if self.sr_ratio > 1:
            if mask is None:
                # Hybrid-scale attention
                # global
                # 2. the multi-head self-attention adopts H independent heads
                # q1: B, N, C -> B, N, C/2
                # reshape: B, N, C/2 -> B, N, num_heads/2, C/2
                # permute: B, N, num_heads/2, C/2 -> B, num_heads/2, N, C/num_heads
                q1 = self.q1(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)

                # permute: B, N, C -> B, C, N
                # reshape: B, C, N -> B, C, H, W
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                # sr: B, C, H, W -> B, C, H/r, W/r
                # reshape: B, C, H/r, W/r -> B, C, H*W/(r*r)
                # permute: B, C, H*W/(r*r) -> B, H*W/(r*r), C
                x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_1 = self.act(self.norm(x_1))
                # kv1: B, H*W/(r*r), C -> B, H*W/(r*r), C
                # reshape: B, H*W/(r*r), C, C -> B, N, C,
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]  # B head N C

                attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # B head Nq Nkv
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)

                # global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) # B Nk  #max ?  mean ?
                # global_mask_value = F.interpolate(global_mask_value.view(B,1,H//self.sr_ratio,W//self.sr_ratio),
                #                                   (H, W), mode='nearest')[:, 0]

                # local
                q2 = self.q2(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1,
                                                                                                3)  # B head N C
                kv2 = self.kv2(x_.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, -1, 2, self.num_heads // 2,
                                                                              C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
                k2, v2 = kv2[0], kv2[1]
                q_window = 7
                window_size = 7
                q2, k2, v2 = window_partition(q2, q_window, H, W), window_partition(k2, window_size, H, W), \
                    window_partition(v2, window_size, H, W)
                attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
                # (B*numheads*num_windows, window_size*window_size, window_size*window_size)
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)

                x2 = (
                            attn2 @ v2)  # B*numheads*num_windows, window_size*window_size, C   .transpose(1, 2).reshape(B, N, C)
                x2 = window_reverse(x2, q_window, H, W, self.num_heads // 2)

                # local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads//2, H//window_size*W//window_size, window_size*window_size, window_size*window_size).mean(1), dim=2)
                # local_mask_value = local_mask_value.view(B, H // window_size, W // window_size, window_size, window_size)
                # local_mask_value=local_mask_value.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)

                # mask B H W
                x = torch.cat([x1, x2], dim=-1)
                x = self.proj(x + lepe)
                x = self.proj_drop(x)
                # cal mask
                # mask = local_mask_value+global_mask_value
                # mask_1 = mask.view(B, H * W)
                # mask_2 = mask.permute(0, 2, 1).reshape(B, H * W)
                # mask = [mask_1, mask_2]
                mask = None
            else:
                # Self-Guided Attention
                q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # mask [local_mask global_mask]  local_mask [value index]  value [B, H, W]
                # use mask to fuse
                mask_1, mask_2 = mask
                mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)
                mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)
                if self.sr_ratio == 8:
                    token1, token2, token3 = H * W // (14 * 14), H * W // 56, H * W // 28
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 4:
                    token1, token2, token3 = H * W // 49, H * W // 14, H * W // 7
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 2:
                    token1, token2 = H * W // 2, H * W // 1
                    token1, token2 = token1 // 2, token2 // 2

                if self.sr_ratio == 4 or self.sr_ratio == 8:
                    p1 = torch.gather(x, 1,
                                      mask_sort_index1[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1,
                                      mask_sort_index1[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3 = torch.gather(x, 1, mask_sort_index1[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)],
                                     dim=-1).permute(0, 2, 1)  # B N C

                    x_ = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(x_, 1,
                                       mask_sort_index2[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1,
                                       mask_sort_index2[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3_.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)],
                                     dim=-1).permute(0, 2, 1)  # B N C
                elif self.sr_ratio == 2:
                    p1 = torch.gather(x, 1,
                                      mask_sort_index1[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)],
                                     dim=-1).permute(0, 2, 1)  # B N C

                    x_ = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(x_, 1,
                                       mask_sort_index2[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)],
                                     dim=-1).permute(0, 2, 1)  # B N C

                kv1 = self.kv1(seq1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                         4)  # kv B heads N C
                kv2 = self.kv2(seq2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x + lepe)
                x = self.proj_drop(x)
                mask = None

        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x + lepe)
            x = self.proj_drop(x)
            mask = None

        return x, mask


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, sr_ratio, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm,
            is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(out_dim, head_count, sr_ratio=sr_ratio)
        self.layer_former_2 = DualTransformerBlock(out_dim, head_count, sr_ratio=sr_ratio)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class SDE_module(nn.Module):
    '''
    Sub-path Direction Excitation Module
    '''

    def __init__(self, in_channels, out_channels, num_class):
        super(SDE_module, self).__init__()
        self.inter_channels = in_channels // 8

        self.att1 = DANetHead(self.inter_channels, self.inter_channels)
        self.att2 = DANetHead(self.inter_channels, self.inter_channels)
        self.att3 = DANetHead(self.inter_channels, self.inter_channels)
        self.att4 = DANetHead(self.inter_channels, self.inter_channels)
        self.att5 = DANetHead(self.inter_channels, self.inter_channels)
        self.att6 = DANetHead(self.inter_channels, self.inter_channels)
        self.att7 = DANetHead(self.inter_channels, self.inter_channels)
        self.att8 = DANetHead(self.inter_channels, self.inter_channels)

        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, out_channels, 1))
        # self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))

        if num_class < 32:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, num_class * 8, 1),
                nn.ReLU(True),
                nn.Conv2d(num_class * 8, in_channels, 1))
        else:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, in_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 1))

    def forward(self, x, d_prior):

        ### re-order encoded_c5 ###
        # new_order = [0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31]
        # # print(encoded_c5.shape)
        # re_order_d_prior = d_prior[:,new_order,:,:]
        # print(d_prior)
        enc_feat = self.reencoder(d_prior)

        ### Channel-wise slicing ###
        feat1 = self.att1(x[:, :self.inter_channels], enc_feat[:, 0:self.inter_channels])
        feat2 = self.att2(x[:, self.inter_channels:2 * self.inter_channels],
                          enc_feat[:, self.inter_channels:2 * self.inter_channels])
        feat3 = self.att3(x[:, 2 * self.inter_channels:3 * self.inter_channels],
                          enc_feat[:, 2 * self.inter_channels:3 * self.inter_channels])
        feat4 = self.att4(x[:, 3 * self.inter_channels:4 * self.inter_channels],
                          enc_feat[:, 3 * self.inter_channels:4 * self.inter_channels])
        feat5 = self.att5(x[:, 4 * self.inter_channels:5 * self.inter_channels],
                          enc_feat[:, 4 * self.inter_channels:5 * self.inter_channels])
        feat6 = self.att6(x[:, 5 * self.inter_channels:6 * self.inter_channels],
                          enc_feat[:, 5 * self.inter_channels:6 * self.inter_channels])
        feat7 = self.att7(x[:, 6 * self.inter_channels:7 * self.inter_channels],
                          enc_feat[:, 6 * self.inter_channels:7 * self.inter_channels])
        feat8 = self.att8(x[:, 7 * self.inter_channels:8 * self.inter_channels],
                          enc_feat[:, 7 * self.inter_channels:8 * self.inter_channels])

        feat = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8], dim=1)

        sasc_output = self.final_conv(feat)
        sasc_output = sasc_output + x

        return sasc_output


class DANetHead(nn.Module):
    '''
    Sub-path excitation (SPE)
    '''

    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        feat_sum = feat_sum * F.sigmoid(enc_feat)
        sasc_output = self.conv8(feat_sum)

        return sasc_output


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # print(self.gamma)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class HSAUnet(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, layers = [[128, 320, 512], [2, 2, 2]]
        sr_ratios = [8, 4, 2, 1]
        heads = [2, 4, 8, 16]
        self.backbone = Encoders(
            image_size=224,
            in_dim=dims,
            heads=heads,
            sr_ratios=sr_ratios,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        # a global attention bottleneck
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count=heads[2],
            sr_ratio=sr_ratios[2],
            token_mlp_mode=token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count=heads[1],
            sr_ratio=sr_ratios[1],
            token_mlp_mode=token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count=heads[1],
            sr_ratio=sr_ratios[1],
            token_mlp_mode=token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        return tmp_0


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.randn((2, 3, 224, 224)).to(device)
    model = HSAUnet(num_classes=2).to(device)
    result = model(images)
    print(f'result shape is {result.shape}')
    # # hybrid_scaled_attention = Attention(64, False, qkv_bias=False, qk_scale=None,attn_drop=0, proj_drop=0, sr_ratio=8)
    # # self_guide_attention = Attention(64, True, qkv_bias=False, qk_scale=None,attn_drop=0, proj_drop=0, sr_ratio=8)
    # patch_sizes = [7, 3, 3, 3]
    # strides = [4, 2, 2, 2]
    # padding_sizes = [3, 1, 1, 1]
    #
    # in_dim = [128, 320, 512]
    # x = torch.randn(2, 3, 224, 224,)
    # patchEmbed = PatchEmbed(in_chans=3, embed_dim=in_dim[0])
    # patched_x, H, W = patchEmbed(x)
    # print(patched_x.shape)
    #
    #
    # # patch_embed
    # # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
    # patch_embed1 = OverlapPatchEmbeddings(
    #     224, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
    # )
    # patched_x, H, W = patch_embed1(x)
    # print(patched_x.shape, H, W)
    # # sgFormer = SGFormer()
    # # multi_organ_segments = sgFormer(x)
    # # print(multi_organ_segments.shape)
    #
    # # x, mask = hybrid_scaled_attention(x, 224, 224, None)
    # # x, mask1 = self_guide_attention(x, 224, 224, mask)
    # # print(x.shape)
