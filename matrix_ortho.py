import numpy as np
import torch

def orthonormalize_weights(w, beta=0.5, iters=20, order=3):
    """
    Script of the Bjork orhtonomalization based on the paper from Bjorck & Al.
    and published in SIAM Journal on Numerical Analysis 8.2 (1971): 358-364. :
    "An iterative algorithm for computing the best estimate of an orthogonal matrix."

    Code is an adaptation from: https://github.com/cemanil/LNets (All credits
    to the original authors)
    """
    if order == 1:
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w = (1 + beta) * w - beta * w.mm(w_t_w)

    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w = (+ (15 / 8) * w
                 - (5 / 4) * w.mm(w_t_w)
                 + (3 / 8) * w.mm(w_t_w_w_t_w))

    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)

            w = (+ (35 / 16) * w
                 - (35 / 16) * w.mm(w_t_w)
                 + (21 / 16) * w.mm(w_t_w_w_t_w)
                 - (5 / 16) * w.mm(w_t_w_w_t_w_w_t_w))


    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w


def scale_values(weight, cuda=True):
    scaler_weight = torch.tensor([np.sqrt(weight.shape[0] * weight.shape[1])]).float()
    return scaler_weight

if __name__ == '__main__':
    pass
