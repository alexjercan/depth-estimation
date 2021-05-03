# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/XinJCheng/CSPN/blob/b3e487bdcdcd8a63333656e69b3268698e543181/cspn_pytorch/utils.py#L19
# - https://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html
#

import torch


class MetricFunction():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.total_size = 0
        self.error_sum = {}
        self.error_avg = {}

    def evaluate(self, predictions, targets):
        depth_p = predictions
        depth_gt = targets
        
        error_val = evaluate_error_depth(depth_p, depth_gt)
        
        self.total_size += self.batch_size
        self.error_avg = avg_error(self.error_sum, error_val, self.total_size, self.batch_size)
        return self.error_avg
    
    def show(self):
        error = self.error_avg
        format_str = ('======DEPTH=======\nMSE=%.4f\tRMSE=%.4f\tMAE=%.4f\tABS_REL=%.4f\nDELTA1.02=%.4f\tDELTA1.05=%.4f\tDELTA1.10=%.4f\nDELTA1.25=%.4f\tDELTA1.25^2=%.4f\tDELTA1.25^3=%.4f')
        return format_str % (error['D_MSE'], error['D_RMSE'], error['D_MAE'],  error['D_ABS_REL'], \
                         error['D_DELTA1.02'], error['D_DELTA1.05'], error['D_DELTA1.10'], \
                         error['D_DELTA1.25'], error['D_DELTA1.25^2'], error['D_DELTA1.25^3'])


def evaluate_error_depth(pred_depth, gt_depth):
    # for numerical stability
    depth_mask = gt_depth>0.0001
    batch_size = gt_depth.size(0)
    error = {}
    _pred_depth = pred_depth[depth_mask]
    _gt_depth   = gt_depth[depth_mask]
    n_valid_element = float(_gt_depth.size(0))

    if n_valid_element > 0:
        n_valid_element = torch.tensor(n_valid_element, device=pred_depth.device)
        diff_mat = torch.abs(_gt_depth-_pred_depth)
        rel_mat = torch.div(diff_mat, _gt_depth)
        error['D_MSE'] = torch.div(torch.sum(torch.pow(diff_mat, 2)), n_valid_element)
        error['D_RMSE'] = torch.sqrt(error['D_MSE'])
        error['D_MAE'] = torch.div(torch.sum(diff_mat), n_valid_element)
        error['D_ABS_REL'] = torch.div(torch.sum(rel_mat), n_valid_element)
        y_over_z = torch.div(_gt_depth, _pred_depth)
        z_over_y = torch.div(_pred_depth, _gt_depth)
        max_ratio = torch.max(y_over_z, z_over_y)
        error['D_DELTA1.02'] = torch.div(torch.sum(max_ratio <= 1.02), n_valid_element)
        error['D_DELTA1.05'] = torch.div(torch.sum(max_ratio <= 1.05), n_valid_element)
        error['D_DELTA1.10'] = torch.div(torch.sum(max_ratio <= 1.10), n_valid_element)
        error['D_DELTA1.25'] = torch.div(torch.sum(max_ratio <= 1.25), n_valid_element)
        error['D_DELTA1.25^2'] = torch.div(torch.sum(max_ratio <= 1.25**2), n_valid_element)
        error['D_DELTA1.25^3'] = torch.div(torch.sum(max_ratio <= 1.25**3), n_valid_element)
    return error


# avg the error
def avg_error(error_sum, error_val, total_size, batch_size):
    error_avg = {}
    for item, value in error_val.items():
        error_sum[item] = error_sum.get(item, 0) + value * batch_size
        error_avg[item] = error_sum[item] / float(total_size)
    return error_avg


def print_single_error(epoch, loss, error):
    format_str = ('%s\nEpoch: %d, loss=%s\n%s\n')
    print (format_str % ('eval_avg_error', epoch, loss, error))