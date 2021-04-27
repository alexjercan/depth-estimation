# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch.nn as nn
from model import RMSELoss


class MetricFunction(nn.Module):
    def __init__(self) -> None:
        super(MetricFunction, self).__init__()
        self.rmse_fn = RMSELoss()
        
        self.rmse_values = []

    def forward(self, predictions, targets):
        (depth_p, normal_p) = predictions
        (depth_gt, normal_gt) = targets
        
        rmse = self.rmse_fn(depth_p, depth_gt)
        self.rmse_values.append(rmse.item())

    def show(self):
        mean_rmse = sum(self.rmse_values) / len(self.rmse_values)
        
        return f'(rmse: {mean_rmse:.4f})'
