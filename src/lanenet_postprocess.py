#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as ops
import math

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


'''Binary Segmentation된 이미지에 대해 Morphology 연산 진행'''
def _morphological_process(image, kernel_size=5):
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneNetCluster(object):

    def __init__(self, cfg):
        self._cfg = cfg


    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret):
        return lane_coordinate

    def apply_lane_feats_cluster(self, binary_seg_result):
        # Binary 이미지의 nonzero 좌표 빼오기
        idx = np.where(binary_seg_result == 255)
        coord = np.vstack((idx[1], idx[0])).transpose()
        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)

        # Segmentation된 라인에 색칠하기
        mask[coord[:, 1], coord[:, 0]] = (255, 255, 255)

        return mask




class LaneNetPostProcessor(object):

    def __init__(self, cfg, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)


    def postprocess(self, binary_seg_result,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)
        
        
        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=3)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0
        
        # apply embedding features cluster
        mask_image = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
        )

        ret = {
            'mask_image': mask_image,
            'source_image': source_image,
        }

        return ret
