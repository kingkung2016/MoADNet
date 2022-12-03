
import torch
from torch import nn
from torch.nn.parameter import Parameter

#----------------------------------------SA Module--------------------------------------------------------
class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


#------------------------------ Inverted Bottleneck Cross-modality Fusion-------------------------------------------------
class IBCMF(nn.Module):
    """Fuse the features from RGB stream and Depth stream.
    Args:
        channel_RGB: the input channels of RGB stream
        channel_Depth: the input channels of Depth stream
        channel_in: the channels after first convolution, makes the channels fed into Feature Fusion are same
        channel_out: the output channels of this module
    """
    def __init__(self, channel_RGB, channel_Depth, channel_in, channel_out):
        super(IBCMF, self).__init__()

        #channel expansion
        self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_in, 1, 1, 0), nn.BatchNorm2d(channel_in), nn.ReLU())
        self.conv_depth = nn.Sequential(nn.Conv2d(channel_Depth, channel_in, 1, 1, 0), nn.BatchNorm2d(channel_in), nn.ReLU())

        #feature fusion
        self.SA_module_rgb = sa_layer(channel_in, groups=channel_in // 2)
        self.SA_module_depth = sa_layer(channel_in, groups=channel_in // 2)

        self.conv_3 = nn.Sequential(nn.Conv2d(channel_in*2, channel_in, 1, 1, 0), nn.BatchNorm2d(channel_in), nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(channel_in, 2, 3, 1, 1), nn.Sigmoid())

        #channel projection
        self.conv_out = nn.Sequential(nn.Conv2d(channel_in, channel_in // 2, 3, 1, 1), nn.ReLU(),
                                      nn.Conv2d(channel_in // 2, channel_in // 4, 3, 1, 1), nn.ReLU(),
                                      nn.Conv2d(channel_in // 4, channel_out, 3, 1, 1), nn.ReLU())

    def forward(self,rgb,depth):
        Fr = self.conv_rgb(rgb)
        Fd = self.conv_depth(depth)

        fusion = torch.cat([Fr, Fd],dim=1)
        fusion = self.conv_3(fusion)
        fusion = self.conv_4(fusion)
        weight_rgb = fusion[:, 0, :, :].unsqueeze(1)
        weight_depth = fusion[:, 1, :, :].unsqueeze(1)

        Fr_out = self.SA_module_rgb(Fr)
        Fd_out = self.SA_module_depth(Fd)

        F_out = Fr_out * weight_rgb + Fd_out * weight_depth + Fr_out * weight_rgb * Fd_out * weight_depth
        F_out = self.conv_out(F_out)

        return F_out

#-----------------------------------------Adaptive Atrous Spatial Pyramid----------------------------------------
# dilated_conv+BN+relu
class Dilated_Conv(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Dilated_Conv, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
			stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class A2SP(nn.Module):
    """Concatenate multiple dilate convolutions in parallel.
    Args:
        inplanes: the input channels
        planes: the output channels
        rates: a list of all dilated_conv rates
    """
    def __init__(self, inplanes, planes, rates):
        super(A2SP, self).__init__()

        #空洞卷积
        self.Dilations = []
        for rate in rates:
            self.Dilations.append(Dilated_Conv(inplanes, planes, rate=rate))
        self.Dilations = nn.Sequential(*self.Dilations)

        #全局平均池化和卷积生成权重
        self.GAP_Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, inplanes * len(rates), 1, stride=1, bias=False),
            nn.Sigmoid()
        )

        #尾部的1x1卷积恢复通道数
        self.conv = nn.Sequential(
            nn.Conv2d(planes*len(rates), planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

    def forward(self, x):
        dilated_out = []
        for dilated_conv in self.Dilations:
            dilated_out.append(dilated_conv(x))

        out = torch.cat(dilated_out,dim=1)

        v_attention = self.GAP_Conv(x)
        y = out * v_attention
        y = self.conv(y)

        return y

#------------------------------------------------Saliency Head--------------------------------------------------------------
class Sal_Head(nn.Module):
    def __init__(self, channel):
        super(Sal_Head, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, 1, 1, 1, 0)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


if __name__ == '__main__':
    #定义网络结构
    test_module = IBCMF(160,160,128,16)

    #查看网络结构
    from torchsummary import summary

    summary(test_module.cuda(), input_size=[(160, 8, 8), (160, 8, 8)])
    print('params: %.2fM' % (sum(p.numel() for p in test_module.parameters())/1000000.0))
