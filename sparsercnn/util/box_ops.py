# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import Tuple
import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def sca(boxes1, boxes2, alpha = 0.5, eps=1e-7):
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    min_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    min_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    min_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    min_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    max_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    max_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    max_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    max_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    min_w = min_x2 - max_x1
    min_h = min_y2 - max_y1
    max_w = max_x2 - min_x1
    max_h = max_y2 - min_y1

    w_ratio = min_w / max_w
    h_ratio = min_h / max_h

    SO = torch.where(100* w_ratio * h_ratio < 0, torch.min(w_ratio, h_ratio), w_ratio + h_ratio)

    return SO

    #d_lt = (boxes1[:, None, 0] - boxes2[:, 0] + eps)**2 + (boxes1[:, None, 1] - boxes2[:, 1] + eps)**2
    #d_rb = (boxes1[:, None, 2] - boxes2[:, 2] + eps)**2 + (boxes1[:, None, 3] - boxes2[:, 3] + eps)**2
    #d_diag = (max_x2 - max_x1 + eps)**2 + (max_y2 - max_y1 + eps)**2
    #CD = (torch.pow(d_lt, 0.5) / (torch.pow(d_diag, 0.5) + eps) + torch.pow(d_rb, 0.5) / (torch.pow(d_diag, 0.5) + eps)) * alpha
    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    #print(diou[0], iou[0], w_ratio[0], h_ratio[0])
    #exit()

    return diou - alpha * SO

def gt_coverage(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter/ area2

#boxes2 should be gt
def complement_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    c_iou = 0.9 * inter / area2 + 0.1 * inter / area1[:, None]
    #c_iou = torch.min(inter / area2, inter / area1[:, None])
    union = area1[:, None] + area2 - inter

    return c_iou, union
    #return c_iou

def generalized_box_ciou(boxes1, boxes2):
    iou, union = complement_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #    _log_api_usage_once(complete_box_iou)

    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)
    
    w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]
    
    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]
    
    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
        
    return diou - alpha * v

def distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #    _log_api_usage_once(distance_box_iou)

    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    diou, _ = _box_diou_iou(boxes1, boxes2, eps=eps)
    return diou

def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tuple[Tensor, Tensor]:
    iou, _ = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps
    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast((x_p[:, None] - x_g[None, :])) ** 2) + (_upcast((y_p[:, None] - y_g[None, :])) ** 2)
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared), iou

def focal_regression(loss, gamma=2.0, eps=1e-6):
    pt = torch.exp(-loss)
    focal_weight = (1 - pt) ** gamma
    focal_loss = -focal_weight * loss.log()

    return focal_loss

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
