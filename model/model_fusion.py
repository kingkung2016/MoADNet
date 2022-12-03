
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from model.attention_module import IBCMF, A2SP, Sal_Head
from model.mobilenet_V3 import mobilenetv3_large, mobilenetv3_small

logger = logging.getLogger(__name__)

class FusionNet(nn.Module):
    def __init__(self, channels=[16, 24, 40, 112, 160]):
        super(FusionNet, self).__init__()

        self.channels = channels
        self.baseline_net = mobilenetv3_large()
        self.depth_net = mobilenetv3_small()

        # Inverted Bottleneck Cross-modality Fusion
        self.IBCMF2 = IBCMF(self.channels[1], self.channels[1], self.channels[0]*8, self.channels[0])
        self.IBCMF3 = IBCMF(self.channels[2], self.channels[2], self.channels[0]*8, self.channels[0])
        self.IBCMF4 = IBCMF(self.channels[3], self.channels[3], self.channels[0]*8, self.channels[0])
        self.IBCMF5 = IBCMF(self.channels[4], self.channels[4], self.channels[0]*8, self.channels[0])

        # Adaptive Atrous Spatial Pyramid
        rates = [[1, 3], [1, 3, 6], [1, 3, 12], [1, 3, 6, 12]]
        self.A2SP5 = A2SP(self.channels[0], self.channels[0], rates[0])
        self.A2SP4 = A2SP(self.channels[0], self.channels[0], rates[1])
        self.A2SP3 = A2SP(self.channels[0], self.channels[0], rates[2])
        self.A2SP2 = A2SP(self.channels[0], self.channels[0], rates[3])

        #训练时额外输出stage2-5的结果
        if self.training:
            self.Sal_Head_2 = Sal_Head(self.channels[0])
            self.Sal_Head_3 = Sal_Head(self.channels[0])
            self.Sal_Head_4 = Sal_Head(self.channels[0])
            self.Sal_Head_5 = Sal_Head(self.channels[0])
        #否则只用最后一个stage输出
        self.Sal_Head_1 = Sal_Head(self.channels[0])

        #初始化
        self.init_weights()
        #加载预训练权重
        self.load_pretrained()

    def forward(self, RGB, depth):
        image_size = RGB.size()[2:]

        #backbone提取特征
        Fr2,Fr3,Fr4,Fr5 = self.baseline_net(RGB)
        Fd2,Fd3,Fd4,Fd5 = self.depth_net(depth)

        # print(h2.shape, h3.shape, h4.shape, h5.shape)
        # print(d2.shape, d3.shape, d4.shape, d5.shape)

        #IBCMF模块
        F2 = self.IBCMF2(Fr2, Fd2)    # 64*64*24
        F3 = self.IBCMF3(Fr3, Fd3)    # 32*32*40
        F4 = self.IBCMF4(Fr4, Fd4)    # 16*16*112
        F5 = self.IBCMF5(Fr5, Fd5)    # 8 *8 *160

        #A2SP模块
        F5_Lin = self.A2SP5(F5)

        F4_Lin = self.A2SP4(F4 + F.interpolate(F5_Lin, F4.shape[2:], mode='bilinear', align_corners=False))

        F3_Lin = self.A2SP3(F3 + F.interpolate(F5_Lin, F3.shape[2:], mode='bilinear', align_corners=False) +
                               F.interpolate(F4_Lin, F3.shape[2:], mode='bilinear', align_corners=False))

        F2_Lin = self.A2SP2(F2 + F.interpolate(F5_Lin, F2.shape[2:], mode='bilinear', align_corners=False) +
                               F.interpolate(F4_Lin, F2.shape[2:], mode='bilinear', align_corners=False) +
                               F.interpolate(F3_Lin, F2.shape[2:], mode='bilinear', align_corners=False))

        #训练阶段输出5个stage
        if self.training:
            #最后做一次3*3反卷积，恢复到原来图片大小并输出结果
            F5_out = F.interpolate(self.Sal_Head_5(F5), image_size, mode='bilinear', align_corners=False)
            F4_out = F.interpolate(self.Sal_Head_4(F5_Lin), image_size, mode='bilinear', align_corners=False)
            F3_out = F.interpolate(self.Sal_Head_3(F4_Lin), image_size, mode='bilinear', align_corners=False)
            F2_out = F.interpolate(self.Sal_Head_2(F3_Lin), image_size, mode='bilinear', align_corners=False)
            F1_out = F.interpolate(self.Sal_Head_1(F2_Lin), image_size, mode='bilinear', align_corners=False)
            return F1_out, F2_out, F3_out, F4_out, F5_out
        else:
            F_out = F.interpolate(self.Sal_Head_1(F2_Lin), image_size, mode='bilinear', align_corners=False)
            return F_out

    def load_pretrained(self):
        #获取当前模型的参数
        baseline_dict = self.baseline_net.state_dict()
        #获取预训练的参数
        pretrained_large_dict = torch.load('./pre-trained/mobilenetv3-large.pth')
        #加载部分能用的参数
        pretrained_large_dict = {k: v for k, v in pretrained_large_dict.items() if k in baseline_dict}
        #print(pretrained_large_dict.keys())
        # 更新现有的model_dict
        baseline_dict.update(pretrained_large_dict)
        # 加载真正需要的state_dict
        self.baseline_net.load_state_dict(baseline_dict)


    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    #定义网络
    channels = [16, 24, 40, 112, 160]
    model_fusion = FusionNet(channels).eval()

    #查看网络结构
    from torchsummary import summary
    summary(model_fusion.cuda(),input_size=[(3,256,256), (3,256,256)])

    #查看输出
    model_fusion_train = FusionNet(channels).train()
    RGB = torch.randn(1,3,256,256)
    depth = torch.randn(1,3,256,256)
    p, p2, p3, p4, p5 = model_fusion_train(RGB, depth)
    print(p.shape, p2.shape, p3.shape, p4.shape, p5.shape)
