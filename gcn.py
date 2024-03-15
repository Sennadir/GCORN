"""
Script implementation of a GCN model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from torch.nn import Parameter
import numpy as np

class ConvClass(nn.Module):
    """
    This is an implementation of the convolution operation in the GCN as
    explained in the original work "Semi-Supervised Classification with Graph
    Convolutional Networks" <https://arxiv.org/abs/1609.02907>
    ---
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation: The activation to be used.
    """
    def __init__(self, input_dim , output_dim, activation):
        super(ConvClass, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.activation = activation
        self.reset_parameters()


    def forward(self, x, adj):
        x = F.linear(x, self.weight)
        return self.activation(torch.mm(adj,x))

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)


class MLPClassifier(nn.Module):
    """
    Implementation of a simple Multi-layer Perceptron that is used after the
    convolution operation to derive the final classification.

    ---
        input_dim (int): The input dimension (usually the last hidden dimension)
        output_dim (int): The output dimension (usually the number of classes)

    """
    def __init__(self, input_dim, output_dim, activation):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.lin(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.activation = nn.ReLU()

        self.conv1 = ConvClass(in_channels, hidden_channels,
                                                activation = self.activation)
        self.conv2 = ConvClass(hidden_channels, hidden_channels,
                                                activation = self.activation)

        self.lin = MLPClassifier(hidden_channels, out_channels, self.activation)


    def forward(self, x, adj, edge_weight=None):
        x = self.conv1(x, adj)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
