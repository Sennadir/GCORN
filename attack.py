"""
Main attack that are used in the evaluation
"""

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import torch.distributions as tdist
from torch.distributions.normal import Normal

class RandomNoise():
    """
    Random Noise attack
    ---
    Budget : Budget of the attack to be generated
    """
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def perturb(self, data):
        x, num_nodes, num_feat = data.x, data.num_nodes, data.num_features

        loc = torch.zeros(num_feat).to(data.x.device)

        normal = Normal(loc, self.noise_ratio / np.sqrt(num_feat))
        noise = normal.sample((num_nodes, ))

        return  noise + data.x




if __name__ == "__main__":
    pass
