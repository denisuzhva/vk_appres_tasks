# model.py

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt



class SincConv(nn.Module):
    """A multi-channel implementation of a 1d sinc convolution layer."""

    @staticmethod
    def f_to_m(f):
        """Convert hz to mel."""
        m = 2595. * torch.log10(1 + f / 700.)
        return m

    @staticmethod
    def m_to_f(m):
        """Convert mel to hz."""
        f = 700. * (torch.pow(10., m / 2595.) - 1)
        return f

    def __init__(self, in_channels, out_channels, kernel_size, fs=16000.):
        super().__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        f_lowest = torch.tensor(0.) # arguably may be bounded by 30 https://www.youtube.com/watch?v=YwnsqhDbcHQ
        f_highest = torch.tensor(fs / 2) 
        m_lowest = self.f_to_m(f_lowest)
        m_highest = self.f_to_m(f_highest)
        self.__window = torch.hamming_window(kernel_size)

        # Random initialization on a mel interval [m_lowest, m_highest]
        m1_abs = torch.rand(out_channels) * (m_highest - m_lowest) / 2 + m_lowest
        m2_abs = m1_abs + torch.abs(torch.rand(out_channels) * (m_highest - m_lowest) / 2 + m_lowest - m1_abs) # f2 = f1 + |f2 - f1|
        f1_abs = self.m_to_f(m1_abs)
        f2_abs = self.m_to_f(m2_abs)
         
        self.__f1 = nn.Parameter(f1_abs / fs)
        self.__fb = nn.Parameter((f2_abs - f1_abs) / fs)
        
        self.__n_space = (torch.arange(kernel_size).float() - (kernel_size - 1) / 2)
                

    def forward(self, x):
        device = x.device
        f1_abs = torch.abs(self.__f1)
        f2_abs = f1_abs + self.__fb
        f1_space = torch.matmul(f1_abs.view(-1, 1), self.__n_space.view(1, -1).to(device))
        f2_space = torch.matmul(f2_abs.view(-1, 1), self.__n_space.view(1, -1).to(device))
        g = 2 * (f2_abs.view(-1, 1).tile((1, self.__kernel_size)) * torch.sinc(2 * np.pi * f2_space) - \
                 f1_abs.view(-1, 1).tile((1, self.__kernel_size)) * torch.sinc(2 * np.pi * f1_space)) * self.__window.to(device)
        #plt.plot(g[0, :].cpu().detach().numpy())
        #plt.show() 
        g_ready = Variable(g.view(self.__out_channels, 1, self.__kernel_size).tile((1, self.__in_channels, 1)))
        out = F.conv1d(x, g_ready)

        return out


class SincNet(nn.Module):
    """CNN with a possible sinc convolution layer at the start."""

    @staticmethod
    def calc_convpool1d_shape(n_filters, data_length_before, conv_kernel_size, pool_kernel_size):
        """
        Calculate new shape after convolution and pooling. 
        Convolution stride should be equal to 1.
        Pooling stride should be equal to the kernel length.
        """
        new_shape = [n_filters, (data_length_before - conv_kernel_size + 1) // pool_kernel_size]
        return new_shape

    def __init__(self, in_channels, in_size, conv_n_filters, 
                 conv_kernel_sizes, pool_kernel_sizes, 
                 fc_sizes, n_classes, 
                 dropout_probs, lrelu_slope=0.01,
                 do_sincconv=False):
        super().__init__()

        shape0 = [in_channels, in_size] 

        # 1st layer modules before general CNN
        if do_sincconv:
            self.__conv1 = SincConv(in_channels, conv_n_filters[0], conv_kernel_sizes[0])
        else:
            self.__conv1 = nn.Conv1d(in_channels, conv_n_filters[0], conv_kernel_sizes[0])
        self.__pool1 = nn.MaxPool1d(pool_kernel_sizes[0])
        shape1 = self.calc_convpool1d_shape(conv_n_filters[0], in_size, conv_kernel_sizes[0], pool_kernel_sizes[0])
        self.__ln1 = nn.LayerNorm(shape1)

        # 2nd layer modules
        self.__conv2 = nn.Conv1d(conv_n_filters[0], conv_n_filters[1], conv_kernel_sizes[1])
        self.__pool2 = nn.MaxPool1d(pool_kernel_sizes[1])
        shape2 = self.calc_convpool1d_shape(conv_n_filters[1], shape1[1], conv_kernel_sizes[1], pool_kernel_sizes[1])
        self.__ln2 = nn.LayerNorm(shape2)

        # 3rd layer modules
        self.__conv3 = nn.Conv1d(conv_n_filters[1], conv_n_filters[2], conv_kernel_sizes[2])
        self.__pool3 = nn.MaxPool1d(pool_kernel_sizes[2])
        shape3 = self.calc_convpool1d_shape(conv_n_filters[2], shape2[1], conv_kernel_sizes[2], pool_kernel_sizes[2])
        self.__ln3 = nn.LayerNorm(shape3)

        # 4th layer modules
        self.__fc4 = nn.Linear(shape3[0] * shape3[1], fc_sizes[0])
        self.__bn4 = nn.BatchNorm1d(fc_sizes[0])
        self.__drop4 = nn.Dropout(dropout_probs[0])

        # 5th layer modules 
        self.__fc5 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.__bn5 = nn.BatchNorm1d(fc_sizes[1])
        self.__drop5 = nn.Dropout(dropout_probs[1])

        # 6th layer modules
        self.__fc6 = nn.Linear(fc_sizes[1], fc_sizes[2])
        self.__bn6 = nn.BatchNorm1d(fc_sizes[2])
        self.__drop6 = nn.Dropout(dropout_probs[2])

        # 7th layer modules
        self.__fc7 = nn.Linear(fc_sizes[2], n_classes)

        # Other modules
        self.__act = nn.LeakyReLU(lrelu_slope)
        self.__softmax = nn.Softmax(dim=1)

    def forward(self, x):

        ## CNN
        # 1st layer
        out = self.__conv1(x)
        out = self.__act(out)
        out = self.__pool1(out)
        out = self.__ln1(out)

        # 2nd layer
        out = self.__conv2(out)
        out = self.__act(out)
        out = self.__pool2(out)
        out = self.__ln2(out)

        # 3rd layer
        out = self.__conv3(out)
        out = self.__act(out)
        out = self.__pool3(out)
        out = self.__ln3(out)

        ## DNN
        # 4th layer
        out = torch.flatten(out, 1)
        out = self.__fc4(out)
        out = self.__bn4(out)
        out = self.__act(out)

        # 5th layer
        out = self.__fc5(out)
        out = self.__bn5(out)
        out = self.__act(out)

        # 6th layer
        out = self.__fc6(out)
        out = self.__bn6(out)
        out = self.__act(out)

        # 7th layer
        out = self.__fc7(out)
        out = self.__act(out)

        # Softmaout 
        out = self.__softmax(out)

        return out