# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/XinJCheng/CSPN/blob/b3e487bdcdcd8a63333656e69b3268698e543181/cspn_pytorch/utils.py#L19
#

import torch


def metric_function(predictions, targets):
    (depth_p, normal_p) = predictions
    (depth_gt, normal_gt) = targets
    
    depth_error = evaluate_error(depth_p, depth_gt)
    return depth_error
    

def max_of_two(y_over_z, z_over_y):
    return torch.max(y_over_z, z_over_y)


def evaluate_error(pred_depth, gt_depth):
    # for numerical stability
    depth_mask = gt_depth>0.0001
    batch_size = gt_depth.size(0)
    error = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
             'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
             'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,
             }
    _pred_depth = pred_depth[depth_mask]
    _gt_depth   = gt_depth[depth_mask]
    n_valid_element = float(_gt_depth.size(0))

    if n_valid_element > 0:
        diff_mat = torch.abs(_gt_depth-_pred_depth)
        rel_mat = torch.div(diff_mat, _gt_depth)
        error['MSE'] = torch.div(torch.sum(torch.pow(diff_mat, 2)), n_valid_element)
        error['RMSE'] = torch.sqrt(error['MSE'])
        error['MAE'] = torch.div(torch.sum(diff_mat), n_valid_element)
        error['ABS_REL'] = torch.div(torch.sum(rel_mat), n_valid_element)
        y_over_z = torch.div(_gt_depth, _pred_depth)
        z_over_y = torch.div(_pred_depth, _gt_depth)
        max_ratio = max_of_two(y_over_z, z_over_y)
        error['DELTA1.02'] = torch.div(torch.sum(max_ratio < 1.02), n_valid_element)
        error['DELTA1.05'] = torch.div(torch.sum(max_ratio < 1.05), n_valid_element)
        error['DELTA1.10'] = torch.div(torch.sum(max_ratio < 1.10), n_valid_element)
        error['DELTA1.25'] = torch.div(torch.sum(max_ratio < 1.25), n_valid_element)
        error['DELTA1.25^2'] = torch.div(torch.sum(max_ratio < 1.25**2), n_valid_element)
        error['DELTA1.25^3'] = torch.div(torch.sum(max_ratio < 1.25**3), n_valid_element)
    return error


# avg the error
def avg_error(error_sum, error_step, total_step, batch_size):
    error_avg = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                 'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                 'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,}
    for item, value in error_step.items():
        error_sum[item] += value * batch_size
        error_avg[item] = error_sum[item] / float(total_step)
    return error_avg


# print error
def print_error(split, epoch, step, loss, error, error_avg, print_out=False):
    format_str = ('%s ===>\n\
                  Epoch: %d, step: %d, loss=%.4f\n\
                  MSE=%.4f(%.4f)\tRMSE=%.4f(%.4f)\tMAE=%.4f(%.4f)\tABS_REL=%.4f(%.4f)\n\
                  DELTA1.02=%.4f(%.4f)\tDELTA1.05=%.4f(%.4f)\tDELTA1.10=%.4f(%.4f)\n\
                  DELTA1.25=%.4f(%.4f)\tDELTA1.25^2=%.4f(%.4f)\tDELTA1.25^3=%.4f(%.4f)\n')
    error_str = format_str % (split, epoch, step, loss,\
                         error['MSE'], error_avg['MSE'], error['RMSE'], error_avg['RMSE'],\
                         error['MAE'], error_avg['MAE'], error['ABS_REL'], error_avg['ABS_REL'],\
                         error['DELTA1.02'], error_avg['DELTA1.02'], \
                         error['DELTA1.05'], error_avg['DELTA1.05'], \
                         error['DELTA1.10'], error_avg['DELTA1.10'], \
                         error['DELTA1.25'], error_avg['DELTA1.25'], \
                         error['DELTA1.25^2'], error_avg['DELTA1.25^2'], \
                         error['DELTA1.25^3'], error_avg['DELTA1.25^3'])
    if print_out:
        print(error_str)
    return error_str


def print_single_error(epoch, step, loss, error):
    format_str = ('%s ===>\n\
                  Epoch: %d, step: %d, loss=%s\n\
                  MSE=%.4f\tRMSE=%.4f\tMAE=%.4f\tABS_REL=%.4f\n\
                  DELTA1.02=%.4f\tDELTA1.05=%.4f\tDELTA1.10=%.4f\n\
                  DELTA1.25=%.4f\tDELTA1.25^2=%.4f\tDELTA1.25^3=%.4f\n')
    print (format_str % ('eval_avg_error', epoch, step, loss,\
                         error['MSE'], error['RMSE'], error['MAE'],  error['ABS_REL'], \
                         error['DELTA1.02'], error['DELTA1.05'], error['DELTA1.10'], \
                         error['DELTA1.25'], error['DELTA1.25^2'], error['DELTA1.25^3']))