import logging
import  torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
import torch.nn.functional as F
from collections import OrderedDict

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels,momentum=0.03, eps=1E-4)
        self.activation = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        # print(x.shape, x.mean(), x.var())
        # exit()
        return x


class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(LinearConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels,momentum=0.03, eps=1E-4)
        # self.activation = nn.LeakyReLU(0.1,inplace=True)
        # self.activation = nn.Linear
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.activation(x)
        # print(x.shape, x.mean(), x.var())
        # exit()
        return x


# 深度可分离卷积
class DepthWiseConvRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthWiseConvRes, self).__init__()
        self.groups_conv = BasicConv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels)
        self.point_conv = LinearConv(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.groups_conv(x)
        x = self.point_conv(x)
        return x


class DepthWiseConvdown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseConvdown, self).__init__()
        self.groups_conv = BasicConv(in_channels, in_channels, kernel_size=3, stride=2,groups=in_channels)
        self.point_conv = LinearConv(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.groups_conv(x)
        x = self.point_conv(x)
        return x


# 残差块
class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()
        self.block = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            DepthWiseConvRes(out_channels, in_channels),
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        x1 = self.block(x)
        return x + x1


# 大残差块
class BigResblock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(BigResblock, self).__init__()
        self.blocks_conv = nn.Sequential(
            *[Resblock(in_channels, out_channels) for _ in range(num_blocks)],
        )

    def forward(self, x):
        x = self.blocks_conv(x)
        return x


# 构建backbone
# layers[1,2,2,4,4,5]
@BACKBONES.register_module()
class YOLOFastestBackBone(nn.Module):
    def __init__(self):
        super(YOLOFastestBackBone, self).__init__()
        self.frozen_stages = -1
        self.norm_eval = False
        self.block1 = nn.Sequential(
                    BasicConv(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                    BasicConv(in_channels=8, out_channels=8, kernel_size=1, stride=1),
                    DepthWiseConvRes(in_channels=8, out_channels=4),

                    BigResblock(in_channels=4, out_channels=8, num_blocks=1),
                    BasicConv(in_channels=4, out_channels=24, kernel_size=1, stride=1),
                    DepthWiseConvdown(in_channels=24, out_channels=8),

                    BigResblock(in_channels=8, out_channels=32, num_blocks=2),
                    BasicConv(in_channels=8, out_channels=32, kernel_size=1, stride=1),
                    DepthWiseConvdown(in_channels=32, out_channels=8),

                    BigResblock(in_channels=8, out_channels=48, num_blocks=2),
                    BasicConv(in_channels=8, out_channels=48, kernel_size=1, stride=1),
                    DepthWiseConvRes(in_channels=48, out_channels=16),

                    BigResblock(in_channels=16, out_channels=96, num_blocks=4),
                    BasicConv(in_channels=16, out_channels=96, kernel_size=1, stride=1),
                    DepthWiseConvdown(in_channels=96, out_channels=24),
                    BigResblock(in_channels=24, out_channels=136, num_blocks=4),
                    BasicConv(in_channels=24, out_channels=136, kernel_size=1, stride=1),
                )
        self.block2 = nn.Sequential(
                    DepthWiseConvdown(in_channels=136, out_channels=48),
                    BigResblock(in_channels=48, out_channels=224, num_blocks=5),
                    BasicConv(in_channels=48, out_channels=96, kernel_size=1, stride=1))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        return tuple([out1, out2])## stride 16, 32

    def _load_model_yolo_u_version_pth(self, pth):
        logger = logging.getLogger()
        logger.info('Loading weights into state dict, name: %s' % (pth))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = self.state_dict()
        pretrained_dict = torch.load(pth, map_location=device)['model']
        pretrained_dict = OrderedDict(pretrained_dict)
        pretrained_weights_list = []
        for k, v in pretrained_dict.items():
            if 'total_params'  in k or 'total_ops' in k:
                continue
            pretrained_weights_list.append([k, v])
        matched_dict = {}
        j = 0
        matched_n = 0
        for key, v in model_dict.items():
            matched = False
            for k in range(j, len(pretrained_weights_list)):
                if pretrained_weights_list[k][1].shape == v.shape:
                    ## matched
                    j = k
                    matched = True
                    matched_n+=1
            if matched:
                matched_dict[key] = pretrained_weights_list[j][1]
        print('matched n: ', matched_n)
        model_dict.update(matched_dict)
        self.load_state_dict(model_dict,strict=False)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
             ##From Wangzhiwei
             self._load_model_yolo_u_version_pth(pretrained)
            #logger = logging.getLogger()
            #load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(YOLOFastestBackBone, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

if __name__ == '__main__':
    import  os

    pretrained = os.path.dirname(__file__)+'/../../..//pretrained/yolo-fastest.pt'
    x = torch.randn((1,3,256,256))
    model = YOLOFastestBackBone()
    model.eval()
    model.init_weights(pretrained=pretrained)
    y = model(x)
    print(y[0].shape, y[1].shape)

