from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from prettytable import PrettyTable

import numpy as np
import torch

def process_metrics(results, classes):
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union, total_area_pred_label, total_area_label)

    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    metrics = dict()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            metrics[key] = val
        else:
            metrics['m' + key] = val

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': classes})
    ret_metrics_class.move_to_end('Class', last=False)
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    print('\n' + class_table_data.get_string())

    return metrics


def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                        num_classes: int, ignore_index: int):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        label (torch.tensor): Ground truth segmentation map
            or label filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_area_to_metrics(total_area_intersect: np.ndarray,
                          total_area_union: np.ndarray,
                          total_area_pred_label: np.ndarray,
                          total_area_label: np.ndarray,
                          metrics: List[str] = ['mIoU'],
                          nan_to_num: Optional[int] = None,
                          beta: int = 1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (np.ndarray): The intersection of prediction
            and ground truth histogram on all classes.
        total_area_union (np.ndarray): The union of prediction and ground
            truth histogram on all classes.
        total_area_pred_label (np.ndarray): The prediction histogram on
            all classes.
        total_area_label (np.ndarray): The ground truth histogram on
            all classes.
        metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
            'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be
            replaced by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
    Returns:
        Dict[str, np.ndarray]: per category evaluation metrics,
            shape (num_classes, ).
    """

    def f_score(precision, recall, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | torch.Tensor): The precision value.
            recall (float | torch.Tensor): The recall value.
            beta (int): Determines the weight of recall in the combined
                score. Default: 1.

        Returns:
            [torch.tensor]: The f-score value.
        """
        score = (1 + beta**2) * (precision * recall) / (
            (beta**2 * precision) + recall)
        return score

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metrics {metrics} is not supported')

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        #elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            #acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            #ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor([
                f_score(x[0], x[1], beta) for x in zip(precision, recall)
            ])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics