#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_back_end.py
# @IDE: PyCharm

import tensorflow as tf

import cnn_basenet

"""
Binary Segmentation 및 Loss 계산에 이용되는 Backend
"""
class LaneNetBackEnd(cnn_basenet.CNNBaseModel):
    def __init__(self, phase, cfg):
        super(LaneNetBackEnd, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._embedding_dims = self._cfg.MODEL.EMBEDDING_FEATS_DIMS
        self._binary_loss_type = self._cfg.SOLVER.LOSS_TYPE

    def _is_net_for_training(self):
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))   

    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        '''
        ont hot 인코딩된 label에 class weights를 곱함
        '''
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        '''
        softmax 알고리즘 적용
        '''
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )

        return loss

    @classmethod
    def _multi_category_focal_loss(cls, onehot_labels, logits, classes_weights, gamma=2.0):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :param gamma:
        :return:
        """
        epsilon = 1.e-7
        alpha = tf.multiply(onehot_labels, classes_weights)
        alpha = tf.cast(alpha, tf.float32)
        gamma = float(gamma)
        y_true = tf.cast(onehot_labels, tf.float32)
        y_pred = tf.nn.softmax(logits, dim=-1)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        
        return loss

    def compute_loss(self, binary_seg_logits, binary_label, name, reuse):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_label_onehot = tf.one_hot(   # 인덱스를 입력받아 one-hot 인코딩 결과를 리턴
                    tf.reshape(
                        tf.cast(binary_label, tf.int32),    # int32형으로 캐스팅
                        shape=[binary_label.get_shape().as_list()[0],
                               binary_label.get_shape().as_list()[1],
                               binary_label.get_shape().as_list()[2]]),
                    depth=self._class_nums, # 차선 또는 차선 아닌것 두개로 구분하기위해 class_nums = 2
                    axis=-1 # 마지막 축
                )

                binary_label_plain = tf.reshape(
                    binary_label,
                    shape=[binary_label.get_shape().as_list()[0] *
                           binary_label.get_shape().as_list()[1] *
                           binary_label.get_shape().as_list()[2] *
                           binary_label.get_shape().as_list()[3]])
                '''
                1차원 binary label에서 각각의 유일한 레이블과 각 인수 크기순서, 레이블 수 리턴
                '''
                unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
                counts = tf.cast(counts, tf.float32)
                inverse_weights = tf.divide(
                    1.0,
                    tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
                )   # 레이블의 각 요소의 평균값에 역수 취함
                if self._binary_loss_type == 'cross_entropy':
                    binary_segmenatation_loss = self._compute_class_weighted_cross_entropy_loss(
                        onehot_labels=binary_label_onehot,
                        logits=binary_seg_logits,
                        classes_weights=inverse_weights
                    )
                
                elif self._binary_loss_type == 'focal': # 판별 모델 loss 계산
                    binary_segmenatation_loss = self._multi_category_focal_loss(
                        onehot_labels=binary_label_onehot,
                        logits=binary_seg_logits,
                        classes_weights=inverse_weights
                    )
                else:
                    raise NotImplementedError

            # calculate class weighted instance seg loss
            
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables(): # 변수 순회
                if 'bn' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))

            l2_reg_loss *= 0.001
            total_loss = binary_segmenatation_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': binary_seg_logits,
                'binary_seg_loss': binary_segmenatation_loss,
            }

        return ret

    def inference(self, binary_seg_logits, name, reuse):
        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with tf.variable_scope(name_or_scope='binary_seg'):
                '''
                softmax를 이용하여 각 레이블을 합쳐 1이 되도록 확률값으로 변환, binary_seg_score에 저장
                '''
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)

                '''
                결과값중 가장 큰 값 가져옴
                '''
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)

        return binary_seg_prediction
