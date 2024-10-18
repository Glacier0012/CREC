# coding=utf-8

import torch


def smooth_L1(y_true, y_pred,sigma=3.0):
    sigma_squared = sigma ** 2

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = y_true - y_pred
    regression_diff = torch.abs(regression_diff)
    regression_loss = torch.where(
        regression_diff<(1.0 / sigma_squared),
        0.5 * sigma_squared * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss.sum()