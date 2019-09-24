from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        """
        input: [*, class_num]
        target: [*, class_num]
        output: [*]
        """
        diff = input - target
        diff_square_sum = np.mean(diff * diff, axis=-1)
        result = 0.5*diff_square_sum
        return result

    def backward(self, input, target):
        '''Your codes here'''
        """
        input: [batch_size, class_num]
        target: [batch_size, class_num]
        output: [class_num, batch_size]
        """
        return (input - target).T


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        softmax_input = np.exp(input)
        softmax_input_sum = np.sum(softmax_input, axis=-1)
        softmax_input = (softmax_input.T / softmax_input_sum).T
        log_softmax_input = np.log(softmax_input)
        return np.sum(-log_softmax_input * target, axis=-1)

    def backward(self, input, target):
        '''Your codes here'''
        """
        input: [batch_size, class_num]
        target: [batch_size, class_num]
        output: [class_num, batch_size]
        """
        softmax_input = np.exp(input)
        softmax_input_sum = np.sum(softmax_input, axis=-1)
        softmax_input = (softmax_input.T / softmax_input_sum).T
        return (softmax_input - target).T
