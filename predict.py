# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import random
import multiprocessing as mp
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm
import torch
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from torch.utils.data import Dataset, DataLoader
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from sparsercnn import SparseRCNNDatasetMapper, add_sparsercnn_config, SparseRCNNWithTTA
from sparsercnn.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from detectron2.data import (DatasetCatalog, MetadataCatalog, MapDataset, get_detection_dataset_dicts )
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data import transforms as T_
from results import gen_blank_res_df, print_res_df, print_update_ic19_res_df, print_tncr_res_df, gen_blank_res_df_tncr, print_icttd_res_df, gen_blank_res_df_gtc, print_gtc_res_df
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from custom_coco_summarize import Summarize
from detectron2.evaluation import inference_on_dataset, COCOEvaluator

def seed_everything(seed = 250):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def register_test_dataset(meta, img_dir, test_dataset_name, test_instances_json):
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    
    image_root = img_dir
    register_coco_instances(name=test_dataset_name, metadata=meta, image_root=image_root, json_file=test_instances_json)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_sparsercnn_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weight_path
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def build_test_loader(test_json_path, test_dataset_name):
    def collate_fn(batch):
        return batch

    test_aug_list = [T_.ResizeShortestEdge([800, 800], 1333)]
    test_dataset = get_detection_dataset_dicts([test_dataset_name], filter_empty = False)
    test_dataset = MapDataset(test_dataset, DatasetMapper(cfg, augmentations=test_aug_list, is_train=False, image_format="RGB"))
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    return test_dataloader

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        required = True,
        help="path to config file",
    )
    parser.add_argument(
        "--input_dir",
        required = True,
        help="The directory path of the input images"
    )
    parser.add_argument(
        "--weight_path",
        required = True,
        help="The path of the pretrained weight"
    )
    parser.add_argument(
        "--gt_json_path",
        required = True,
        help="The path of the ground truth json file"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


def predict(model, test_dataloader, test_json_path):
    #results_df = gen_blank_res_df()
    #results_df = gen_blank_res_row_col_df()
    results_df = gen_blank_res_df_tncr()
    #results_df = gen_blank_res_df_gtc()
    results = []
    with torch.no_grad():
        model.eval()
        for idx, items in enumerate(test_dataloader):
            outputs = model(items)
            boxes = outputs[0]["instances"].get_fields()["pred_boxes"].tensor.detach().cpu().numpy()
            scores = outputs[0]["instances"].get_fields()["scores"].detach().cpu().numpy()
            classes = outputs[0]["instances"].get_fields()["pred_classes"].detach().cpu().numpy()
            for idx, bbox in enumerate(boxes):
                bbox = list(bbox)
                result = {
                        "image_id": items[0]["image_id"],
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "category_id": classes[idx] + 1,
                        "score": scores[idx]
                        }
                results.append(result)
        
        if len(results) > 0:
            #Coco eval
            coco_gt = COCO(test_json_path)
            coco_dt = coco_gt.loadRes(results)

            annType = 'bbox'
            coco_eval = COCOeval(coco_gt, coco_dt, annType)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            #For result_df
            summary = Summarize(coco_eval.stats, coco_eval.params, coco_eval.eval)
            #Returns an array of size 20
            #summary_dets = summary.summarizeDets()
            #summary_dets = summary.summarizeDets()
            summary_dets = summary.summarizeDetsTNCR()
            #summary_dets = summary.summarizeDetsGTC()
            results_df.loc[len(results_df)] = summary_dets
        else:
            print("No results yet")
            results_df.loc[len(results_df)] = [0] * 16
    #print_res_df(results_df)
    #print_res_df(results_df)
    print_tncr_res_df(results_df)
    #print_gtc_res_df(results_df)
    #print_update_ic19_res_df(results_df, combined = False)

    return results_df


if __name__ == "__main__":
    args = get_parser().parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = {"ting_classes": ['table']}
    #img_dir = "/data/datasets/hugging_face_td_dataset/open_tables_icttd_for_table_detection/Merged/images"
    img_dir = args.input_dir
    #test_json_path = "/data/datasets/hugging_face_td_dataset/open_tables_icttd_for_table_detection/Merged/merged_test.json"
    test_json_path = args.gt_json_path
    register_test_dataset(meta = meta, img_dir = img_dir, test_dataset_name = "td_test_set", test_instances_json = test_json_path)
    
    cfg = setup_cfg(args)
    model = build_model(cfg)
    weight_list = [args.weight_path]

    #weight_list = glob.glob("/data/logs/icttd_opentable_merged_300/model_final.pth")
    #weight_list = sorted(weight_list)

    result_list = []
    for weight_path in tqdm(weight_list):
        DetectionCheckpointer(model).load(weight_path)
        model.to(device)
        
        test_loader = build_test_loader(test_json_path, test_dataset_name = "td_test_set")
        #evaluator = COCOEvaluator("test_ic19")
        #print(inference_on_dataset(model, test_loader, evaluator))
        results_df = predict(model, test_loader, test_json_path)
        result_list.append(results_df)

    for weight_path, redf in zip(weight_list, result_list):
        print_icttd_res_df(redf, combined = False)

#python predict.py --input_dir "/data/datasets/hugging_face_td_dataset/open_tables_icttd_for_table_detection/Merged/images" --gt_json_path "/data/datasets/hugging_face_td_dataset/open_tables_icttd_for_table_detection/Merged/merged_test.json" --config-file "configs/icttd_opentables.res50.300pro.yaml" --weight_path "/data/logs/icttd_opentable_merged_300/model_final.pth"
