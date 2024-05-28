# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# adapted from OFA: https://github.com/mit-han-lab/once-for-all
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.nn_base import MyNetwork
import kornia

class AttentiveNasStaticModel(MyNetwork):

    def __init__(self, first_conv, blocks, last_conv, classifier, resolution, use_v3_head=True, ch_out_blocks=None, ch_index=None, output_size=[480, 640]):
        super(AttentiveNasStaticModel, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = last_conv

        self.resolution = resolution #input size

        self.ch_out_blocks = ch_out_blocks
        self.ch_index = ch_index

        self.decode_output0 = self.ch_out_blocks[0]
        self.decode_output1 = self.ch_out_blocks[1]
        self.decode_output2 = self.ch_out_blocks[2]
        self.decode_output3 = self.ch_out_blocks[3]
        self.decode_output4 = self.ch_out_blocks[4]
        self.decode_output5 = self.ch_out_blocks[5]
        self.decode_output6 = self.ch_out_blocks[6]

        self.output_size = output_size
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

    def forward(self, x):
        # resize input to target resolution first
        # if x.size(-1) != self.resolution:
        #     x = torch.nn.functional.interpolate(x, size=self.resolution, mode='bicubic', align_corners=True)

        if x.size(-1) != self.resolution:
            if self.resolution == 384:
                x = F.interpolate(x, size=(288, self.resolution), mode='bicubic', align_corners=True)
            elif self.resolution == 256:
                x = F.interpolate(x, size=(192, self.resolution), mode='bicubic', align_corners=True)

        x = self.first_conv(x)
        out1 = x

        i = 0
        #7 5 3 2 2 1
        for block in self.blocks:
            x = block(x)
            if i == self.ch_index[1]:
                out2 = x
            elif i ==self.ch_index[2]:
                out3 = x
            elif i == self.ch_index[4]:
                out4 = x
            i = i + 1

        x = self.last_conv(x)
        out5 = x

        x0 = self.decode_output0(out5)

        x1 = torch.cat((x0,out4), 1)
        x1 = self.decode_output1(x1)

        x2 = torch.cat((x1,out3), 1)
        x2 = self.decode_output2(x2)

        x3 = torch.cat((x2, out2), 1)
        x3 = self.decode_output3(x3)

        x4 = torch.cat((x3, out1), 1)
        x4 = self.decode_output4(x4)

        x = self.decode_output5(x4)

        x = self.decode_output6(x)

        x = F.interpolate(x, size=(self.output_size[0],self.output_size[1]), mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)

        x = x.squeeze(1)
        return x


    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        #_str += self.last_conv.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': AttentiveNasStaticModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'last_conv': self.last_conv.config,
            'classifier': self.classifier.config,
            'resolution': self.resolution
        }


    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()