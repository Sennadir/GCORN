'''
    This implementation have been extracted and adapted from the original
    Parseval implementation:
        https://github.com/mathialo/parsnet/tree/master/parsnet
    ----
    Note that the original implementation was used for image-based models and
    hence the adaptation was done to the context of GCNs.
'''

import torch

def parseval_weight_projections(model_temp, scale_param, num_passes=5):
    # First Convolution
    param = model_temp.conv1.weight.data
    last = param
    for i in range(num_passes):
        temp1 = torch.mm(param.t(), param)
        temp2 = (1 + scale_param) * param - scale_param *  torch.mm(param, temp1)
        last = temp2

    model_temp.conv1.weight.data = last

    # Second Convolution
    param = model_temp.conv2.weight.data
    last = param
    for i in range(num_passes):
        temp1 = torch.mm(param.t(), param)
        temp2 = (1 + scale_param) * param - scale_param *  torch.mm(param, temp1)
        last = temp2

    model_temp.conv2.weight.data = last

    # Linear ReadOut
    param = model_temp.lin.lin.weight.data
    last = param
    for i in range(num_passes):
        temp1 = torch.mm(param.t(), param)
        temp2 = (1 + scale_param) * param - scale_param *  torch.mm(param, temp1)
        last = temp2

    model_temp.lin.lin.weight.data = last

    return model_temp
