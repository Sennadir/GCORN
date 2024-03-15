"""
Script implementation of the GCORN model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from torch.nn import Parameter
import numpy as np
from matrix_ortho import *

class ConvClass(nn.Module):
    """
    This is an adaptation of the original implementation of the GCN to take into
    account the orthogonal projection (as explained in the paper).
    ---
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation: The activation to be used.
        iteration_val (int) : The number of projection iteration
        order_val (int) : The order of the projection
    """

    def __init__(self, input_dim , output_dim, activation, beta_val=0.5,
                iteration_val=25, order_val=2):
        super(ConvClass, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim))

        self.activation = activation

        self.reset_parameters()

        self.beta_val = beta_val
        self.iters = iteration_val
        self.order = order_val

    def forward(self, x, adj):
        scaling = scale_values(self.weight.data).to(x.device)
        ortho_w = orthonormalize_weights(self.weight.t() / scaling,
                                        beta = self.beta_val,
                                        iters = self.iters,
                                        order = self.order).t()

        x = F.linear(x, ortho_w)

        return self.activation(torch.mm(adj,x))

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.lin(x)
        return x


class GCORN(torch.nn.Module):
    """
    Class implementation of an adapted 2-Layers GCN (GCORN) as explained in the
    paper. The model consists of two GCORN propagation with a final MLP layer
    as a ReadOut.
    ---
    in_channels (int) : The input dimension
    hidden_channels (int) : The hidden dimension of the embeddings to be used
    out_channels (int) : The output dimension (the number of classes to predict)
    """
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
