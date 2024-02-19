#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .static_loss import SetCriterion, HungarianMatcher
from .dynamic_loss import SetCriterionDynamicK, HungarianMatcherDynamicK

from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

__all__ = ["SparseRCNN"]

@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        
        self.noise_var = cfg.MODEL.SparseRCNN.NOISE_VAR
        self.noise_mean = cfg.MODEL.SparseRCNN.NOISE_MEAN

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        #max_value = math.sqrt(1/self.hidden_dim)
        #min_value = -math.sqrt(1/self.hidden_dim)
        #self.init_proposal_features = (max_value - min_value) * torch.rand((self.num_proposals, self.hidden_dim), device = self.device) + min_value

        #self.init_proposal_features = torch.rand((self.num_proposals, self.hidden_dim), device = self.device)
        self.box_init_method = cfg.MODEL.SparseRCNN.BOX_INIT_METHOD

        #self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        #nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        #nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        self.use_nms = cfg.MODEL.SparseRCNN.USE_NMS
        self.nms_threshold = cfg.MODEL.SparseRCNN.NMS_THRESH

        # Build Criterion.
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes"]

        if cfg.MODEL.SparseRCNN.CRITERION_DYNAMIC:
            matcher = HungarianMatcherDynamicK(cfg=cfg,
                    cost_class=class_weight, 
                    cost_bbox=l1_weight, 
                    cost_giou=giou_weight,
                    use_focal=self.use_focal)
        
            self.criterion = SetCriterionDynamicK(cfg=cfg,
                    num_classes=self.num_classes,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    eos_coef=no_object_weight,
                    losses=losses,
                    use_focal=self.use_focal)
        else:
            matcher = HungarianMatcher(cfg=cfg,
                    cost_class=class_weight, 
                    cost_bbox=l1_weight, 
                    cost_giou=giou_weight,
                    use_focal=self.use_focal)

            self.criterion = SetCriterion(cfg=cfg,
                    num_classes=self.num_classes,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    eos_coef=no_object_weight,
                    losses=losses,
                    use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def init_proposals_image_size(self, mean = 0, variance = 0.1):
        x = torch.ones(self.num_proposals, 4, device = self.device)
        x[:, :2] = x[:, :2] * 0.5

        noise = mean + torch.randn((self.num_proposals, 2), device = self.device) * variance
        noise = torch.clamp(noise, min=-0.5, max=0.5)
        x = x + torch.concat((noise, -2 * torch.abs(noise)), dim = 1)

        return x

    def init_proposals_random(self, mean = 0.5, variance = 0.5):
        scale = 2
        x = mean + torch.randn((self.num_proposals, 4), device = self.device) * variance
        x = torch.clamp(x, min=-1 * scale, max=scale)
        x = ((x / scale) + 1) / 2.

        return x

    def init_proposals_concat(self, batched_inputs, is_train):

        scale = 2
        if is_train:
            targets = [x["instances"].to(self.device) for x in batched_inputs]
            init_proposals = []
            for targets_per_image in targets:
                h, w = targets_per_image.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                num_gt = gt_boxes.shape[0]
                if num_gt == 0:
                    gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
                    num_gt = 1

                num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
                repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (self.num_proposals % num_gt)
                assert sum(repeat_tensor) == self.num_proposals
                random.shuffle(repeat_tensor)
                repeat_tensor = torch.tensor(repeat_tensor, device=self.device)
                x = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)
                x = x + torch.randn_like(x)
                x = torch.clamp(x, min=-1 * scale, max=scale)
                x = ((x / scale) + 1) / 2.
                x = box_cxcywh_to_xyxy(x)

                x = torch.ones(self.num_proposals, 4, device=self.device)
                init_proposals.append(x)

            init_proposals = torch.stack(init_proposals)
            return init_proposals
        else:
            image_list = [x["image"].to(self.device) for x in batched_inputs]

            mean = 0.5
            variance = 0.5
            bs = len(batched_inputs)
            x = mean + torch.randn((bs, self.num_proposals, 4), device = self.device) * variance
            x = torch.clamp(x, min=-1 * scale, max=scale)
            x = ((x / scale) + 1) / 2.
            x = box_cxcywh_to_xyxy(x)

            return x

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        #proposal_boxes = self.init_proposal_boxes.weight.clone()
        #proposal_boxes = self.init_proposal_boxes.weight
        #proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        #proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]


        if self.box_init_method == "Image":
            proposal_boxes = self.init_proposals_image_size(mean = self.noise_mean, variance = self.noise_var)
        elif self.box_init_method == "Random":
            proposal_boxes = self.init_proposals_random()
        else:
            raise("no impelmentation")

        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        #outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b }
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]

            results = self.inference(box_cls, box_pred, images.image_sizes)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
            

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, box_pred, image_sizes)):

                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, self.nms_threshold)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, self.nms_threshold)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
