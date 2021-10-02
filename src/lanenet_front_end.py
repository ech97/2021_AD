#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午3:53
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_front_end.py
# @IDE: PyCharm

import cnn_basenet
#import vgg16_based_fcn
import bisenet_v2


class LaneNetFrondEnd(cnn_basenet.CNNBaseModel):
    def __init__(self, phase, net_flag, cfg):
        super(LaneNetFrondEnd, self).__init__()
        self._cfg = cfg

        '''
        Segmentation을 위한 Bisenet 인스턴스하여 net 멤버변수에 저장
        '''
        self._frontend_net_map = {
            'bisenetv2': bisenet_v2.BiseNetV2(phase=phase, cfg=self._cfg),
        }

        self._net = self._frontend_net_map[net_flag]

    def build_model(self, input_tensor, name, reuse):

        return self._net.build_model(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse
        )
