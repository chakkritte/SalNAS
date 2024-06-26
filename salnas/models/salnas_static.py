# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch.nn as nn
import kornia
import torch.nn.functional as F
import torch

from ofa.utils.layers import (
    set_layer_from_config,
    MBConvLayer,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    ResidualBlock,
)
from ofa.utils import MyNetwork, make_divisible, MyGlobalAvgPool2d

__all__ = ["MobileNetV3"]


class MobileNetV3(MyNetwork):
    def __init__(
        self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
    ):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def set_decode(self, decode_output, ch_index, output_size):
        self.decode_output = decode_output
        self.ch_index = ch_index
        self.output_size = output_size
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

    def forward(self, x):
        x = self.first_conv(x)
        out1 = x
        
        i = 0
        for block in self.blocks:
            x = block(x)
            if i == self.ch_index[0]:
                out2 = x
            elif i ==self.ch_index[1]:
                out3 = x
            elif i == self.ch_index[3]:
                out4 = x
            i = i + 1

        x = self.final_expand_layer(x)
        out5 = x

        x0 = self.decode_output[0](out5)
        x0 = F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True)

        x1 = torch.cat((x0,out4), 1)
        x1 = self.decode_output[1](x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        x2 = torch.cat((x1,out3), 1)
        x2 = self.decode_output[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)

        x3 = torch.cat((x2, out2), 1)
        x3 = self.decode_output[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)

        x4 = torch.cat((x3, out1), 1)
        x4 = self.decode_output[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.decode_output[5](x4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.decode_output[6](x)

        x = F.interpolate(x, size=(self.output_size[0],self.output_size[1]), mode='bilinear', align_corners=False)

        if not self.training:
            x = self.blur(x)

        x = x.squeeze(1)
        return x
    
        # for decode_block in self.decode_output:

        # x = self.global_avg_pool(x)  # global average pooling
        # x = self.feature_mix_layer(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.final_expand_layer.module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": MobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "final_expand_layer": self.final_expand_layer.config,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        final_expand_layer = set_layer_from_config(config["final_expand_layer"])
        feature_mix_layer = set_layer_from_config(config["feature_mix_layer"])
        classifier = set_layer_from_config(config["classifier"])

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(ResidualBlock.build_from_config(block_config))

        net = MobileNetV3(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBConvLayer) and isinstance(
                    m.shortcut, IdentityLayer
                ):
                    m.conv.point_linear.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks[1:], 1):
            if block.shortcut is None and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            use_bn=True,
            act_func="h_swish",
            ops_order="weight_bn_act",
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for (
                k,
                mid_channel,
                out_channel,
                use_se,
                act_func,
                stride,
                expand_ratio,
            ) in block_config_list:
                mb_conv = MBConvLayer(
                    feature_dim,
                    out_channel,
                    k,
                    stride,
                    expand_ratio,
                    mid_channel,
                    act_func,
                    use_se,
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim,
            feature_dim * 6,
            kernel_size=1,
            use_bn=True,
            act_func="h_swish",
            ops_order="weight_bn_act",
        )
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim * 6,
            last_channel,
            kernel_size=1,
            bias=False,
            use_bn=False,
            act_func="h_swish",
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(
        cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None
    ):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != "0":
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != "0":
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != "0":
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [
                    copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)
                ]
                cfg[stage_id] = new_block_config_list
        return cfg

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()

        for key in state_dict:
            if key not in current_state_dict:
                assert ".mobile_inverted_conv." in key
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super(MobileNetV3, self).load_state_dict(current_state_dict)

    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()