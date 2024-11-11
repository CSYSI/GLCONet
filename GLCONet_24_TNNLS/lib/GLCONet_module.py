import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from lib.GatedConv import GatedConv2dWithActivation
from einops import rearrange
import numbers


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv_7 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=7, stride=1, padding=3,
                                  groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features*3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.dwconv_3(x).chunk(2, dim=1)
        x_3 = F.gelu(x1_3) * x2_3
        x1_5, x2_5 = self.dwconv_5(x).chunk(2, dim=1)
        x_5 = F.gelu(x1_5) * x2_5
        x1_7, x2_7 = self.dwconv_7(x).chunk(2, dim=1)
        x_7 = F.gelu(x1_7) * x2_7

        x = self.project_out(torch.cat((x_3,x_5,x_7),1))
        return x

    def initialize(self):
        weight_init(self)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.qkv1conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv2conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv3conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_3 = self.qkv1conv_3(self.qkv_0(x))
        k_3 = self.qkv2conv_3(self.qkv_1(x))
        v_3 = self.qkv3conv_3(self.qkv_2(x))

        q_3 = rearrange(q_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_3 = rearrange(k_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_3 = rearrange(v_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_3 = torch.nn.functional.normalize(q_3, dim=-1)
        k_3 = torch.nn.functional.normalize(k_3, dim=-1)
        attn_3 = (q_3 @ k_3.transpose(-2, -1)) * self.temperature
        attn_3 = attn_3.softmax(dim=-1)
        out_3 = (attn_3 @ v_3)
        out_3 = rearrange(out_3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_5 = self.qkv1conv_5(self.qkv_0(x))
        k_5 = self.qkv2conv_5(self.qkv_1(x))
        v_5 = self.qkv3conv_5(self.qkv_2(x))

        q_5 = rearrange(q_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_5 = rearrange(k_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_5 = rearrange(v_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_5 = torch.nn.functional.normalize(q_5, dim=-1)
        k_5 = torch.nn.functional.normalize(k_5, dim=-1)
        attn_5 = (q_5 @ k_5.transpose(-2, -1)) * self.temperature
        attn_5 = attn_5.softmax(dim=-1)
        out_5 = (attn_5 @ v_5)
        out_5 = rearrange(out_5, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_7 = self.qkv1conv_7(self.qkv_0(x))
        k_7 = self.qkv2conv_7(self.qkv_1(x))
        v_7 = self.qkv3conv_7(self.qkv_2(x))

        q_7 = rearrange(q_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_7 = rearrange(k_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_7 = rearrange(v_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_7 = torch.nn.functional.normalize(q_7, dim=-1)
        k_7 = torch.nn.functional.normalize(k_7, dim=-1)
        attn_7 = (q_7 @ k_7.transpose(-2, -1)) * self.temperature
        attn_7 = attn_7.softmax(dim=-1)
        out_7 = (attn_7 @ v_7)
        out_7 = rearrange(out_7, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat((out_3,out_5,out_7),1))
        return out



    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module): # Multi-scale transformer block
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class Global(nn.Module): # Global perception module
    def __init__(self, in_channel, out_channel):
        super(Global, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=1,dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3, 1, 1),nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.relu = nn.ReLU(True)
        self.Global = MSA_head(dim=out_channel)



    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x_global = self.Global(x1)
        x_res  = self.res(x)
        x    = self.reduce(torch.cat((x_res,x_global),1)) + x0
        return x


class Global_1(nn.Module):  #MTB in decoder
    def __init__(self, in_channel, out_channel):
        super(Global_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=1,dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3, 1, 1),nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channel, out_channel//2, 3, padding=1),nn.BatchNorm2d(out_channel//2),nn.PReLU(), nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channel//2, 1, 1)
        )

        self.Global = MSA_head(dim=out_channel)



    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x_global = self.Global(x1)
        x_res  = self.res(x)
        x    = self.reduce(torch.cat((x_res,x_global),1)) + x0
        x    = self.out(x)

        return x


class Local(nn.Module): # Local refinement module # Progressive convolution block
    def __init__(self, in_channel, out_channel):
        super(Local, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=1,dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.dilate3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=3,dilation=3),nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel*4, 1),nn.BatchNorm2d(out_channel*4),
            nn.Conv2d(out_channel*4, out_channel, 1),nn.BatchNorm2d(out_channel),
        )

        self.dilate5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5),nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * 4, 1),nn.BatchNorm2d(out_channel*4),
            nn.Conv2d(out_channel * 4, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.dilate7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7),nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * 4, 1),nn.BatchNorm2d(out_channel*4),
            nn.Conv2d(out_channel * 4, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.DConv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1,groups=out_channel),nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * 4, 1),nn.BatchNorm2d(out_channel*4),
            nn.Conv2d(out_channel * 4, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.DConv5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups=out_channel),nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * 4, 1),nn.BatchNorm2d(out_channel*4),
            nn.Conv2d(out_channel * 4, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.DConv7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1, groups=out_channel),nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * 4, 1),nn.BatchNorm2d(out_channel*4),
            nn.Conv2d(out_channel * 4, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.reduce1 = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, 1, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 3, 1, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3,1,1),nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x1_3 = self.dilate3(x1)
        x1_5 = self.dilate5(x1)
        x1_7 = self.dilate7(x1)
        x_1  = self.reduce1(torch.cat((x1,x1_3,x1_5),1))
        x_2  = self.reduce1(torch.cat((x1,x1_3,x1_7),1))
        x_3  = torch.add(self.reduce1(torch.cat((x_1,x_2,x1_3),1)),x1)

        x2_3 = self.DConv3(x_3)
        x2_5 = self.DConv5(x_3)
        x2_7 = self.DConv7(x_3)
        x_2_1 = self.reduce1(torch.cat((x_3,x2_3,x2_5),1))
        x_2_2 = self.reduce1(torch.cat((x_3,x2_3,x2_7),1))
        x_local = torch.add(self.reduce1(torch.cat((x2_3,x_2_1,x_2_2),1)),x_3)

        x_res  = self.res(x)
        x    = self.reduce(torch.cat((x_local,x_res),1))+x0

        return x


class GL_FI(nn.Module): #Group-wise hybrid interaction module
    def __init__(self, in_channels=128):
        super(GL_FI, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(in_channels // 4), nn.ReLU(True)
        )

        self.gatedconv = GatedConv2dWithActivation(in_channels, in_channels, kernel_size=3, stride=1,
                                                   padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
                                                   activation=torch.nn.LeakyReLU(0.2, inplace=True))
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )

    def forward(self, G, L):
        G_1, G_2, G_3, G_4 = self.conv_1(G).chunk(4, dim=1)
        L_1, L_2, L_3, L_4 = self.conv_1(L).chunk(4, dim=1)
        F_1 = self.conv_3(torch.add(G_1, L_1))
        F_2 = self.conv_3(torch.add(G_2, L_2))
        F_3 = self.conv_3(torch.add(G_3, L_3))
        F_4 = self.conv_3(torch.add(G_4, L_4))
        FI = torch.cat((F_1, F_2, F_3, F_4), 1)
        FI = self.reduce(torch.cat((G, L, FI), 1))
        FI = self.conv_1(self.gatedconv(FI) * FI + FI)

        return FI

class FI_1(nn.Module): #Adjacent reverse decoder
    def __init__(self, in_channels, mid_channels):
        super(FI_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.GL_FI = GL_FI(in_channels)

    def forward(self, G, L, prior_cam):

        GL = self.GL_FI(G, L)

        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear', align_corners=True)

        yt = self.conv(torch.cat([GL, prior_cam.expand(-1, L.size()[1], -1, -1)], dim=1))

        conv_out = self.conv3(yt)

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        ra_out = r_prior_cam.expand(-1, L.size()[1], -1, -1).mul(GL)

        cat_out = torch.cat([ra_out, conv_out], dim=1)

        y = self.out_y(cat_out)

        y = y + prior_cam
        return y



class FI_2(nn.Module): #  Adjacent reverse decoder
    def __init__(self, in_channels, mid_channels):
        super(FI_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        self.GL_FI = GL_FI(in_channels)

    def forward(self, G, L, x1, prior_cam):
        GL = self.GL_FI(G, L)

        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=L.size()[2:], mode='bilinear', align_corners=True)

        yt = self.conv(torch.cat([GL, prior_cam.expand(-1, L.size()[1], -1, -1), x1_prior_cam.expand(-1, L.size()[1], -1, -1)],dim=1))
        conv_out = self.conv3(yt)

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        r1_prior_cam = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r_prior_cam = r_prior_cam + r1_prior_cam
        ra_out = r_prior_cam.expand(-1, L.size()[1], -1, -1).mul(GL)

        cat_out = torch.cat([ra_out, conv_out], dim=1)

        y = self.out_y(cat_out)
        y = y + prior_cam + x1_prior_cam
        return y




