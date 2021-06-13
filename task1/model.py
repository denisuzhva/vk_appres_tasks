# model.py

import numpy as np
import torch
from torch import nn
from sinc_conv import SincConv



class SincNet(nn.Module):
    """CNN with a possible sinc convolution layer at the start."""

    def __init__(self, in_channels, in_size, conv_channels, 
                 conv_kernel_sizes, pool_kernel_sizes, 
                 fc_sizes, n_classes, 
                 dropout_probs, lrelu_slope=0.2,
                 do_sincconv=False):
        super(SincNet, self).__init__()

        # 1st layer modules before general CNN
        if do_sincconv:
            self.__conv1 = None
        else:
            self.__conv1 = nn.Conv1d(in_channels, conv_channels[0], conv_kernel_sizes[0])
        self.__pool1 = nn.MaxPool1d(pool_kernel_sizes[0])
        self.__gnorm1 = nn.GroupNorm(1, conv_channels[0])
        self.__drop1 = nn.Dropout(dropout_probs[0])

        # 2nd layer modules
        self.__conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], conv_kernel_sizes[1])
        self.__pool2 = nn.MaxPool1d(pool_kernel_sizes[1])
        self.__gnorm2 = nn.GroupNorm(1, conv_channels[1])

        # 3rd layer modules
        self.__conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], conv_kernel_sizes[2])
        self.__pool3 = nn.MaxPool1d(pool_kernel_sizes[2])
        self.__gnorm3 = nn.GroupNorm(1, conv_channels[2])

        # 4th layer modules
        fc_in_dim = conv_channels[-1] * (((in_size - conv_kernel_sizes[0] + 1) // pool_kernel_sizes[0] - \
                                         conv_kernel_sizes[1] + 1) // pool_kernel_sizes[1] - \
                                         conv_kernel_sizes[2] + 1) // pool_kernel_sizes[2]
        self.__fc4 = nn.Linear(fc_in_dim, fc_sizes[0])
        self.__drop4 = nn.Dropout(dropout_probs[1])

        # 5th layer modules 
        self.__fc5 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.__drop5 = nn.Dropout(dropout_probs[2])

        # 6th layer modules
        self.__fc6 = nn.Linear(fc_sizes[1], n_classes)

        # Other modules
        self.__lrelu = nn.LeakyReLU(lrelu_slope)
        self.__softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # 1st layer
        x = self.__drop1(self.__lrelu(self.__gnorm1(self.__pool1(self.__conv1(x)))))

        # 2nd layer
        x = self.__lrelu(self.__gnorm2(self.__pool2(self.__conv2(x))))

        # 3rd layer
        x = self.__lrelu(self.__gnorm3(self.__pool3(self.__conv3(x))))

        # 4th layer
        x = torch.flatten(x, 1)
        x = self.__drop4(self.__lrelu(self.__fc4(x)))

        # 5th layer
        x = self.__drop5(self.__lrelu(self.__fc5(x)))

        # 6th layer
        x = self.__softmax(self.__lrelu(self.__fc6(x)))

        return x