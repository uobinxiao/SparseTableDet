# Table detection for visually rich document images

## Paper Link
https://www.sciencedirect.com/science/article/abs/pii/S0950705123008304 \
https://arxiv.org/abs/2305.19181

## Dataset
You can download the table detection dataset with the following link:
```
https://huggingface.co/datasets/uobinxiao/open_tables_icttd_for_table_detection
```

## Requirements
This codebase is built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [Sparse-RCNN](https://github.com/PeizeSun/SparseR-CNN/tree/main).
Follow the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install Detectron2.

## Configuration and Training
Set config.yaml based on the dataset.\
Set configs/icttd_opentables.res50.300pro.yaml to modify the model parameters and the output log directory.\
Use python train_net.py to train the model.

## Evaluation
Use python predict.py to evaluate the model.\
One command example: \
python predict.py --input_dir "/data/datasets/hugging_face_td_dataset/open_tables_icttd_for_table_detection/Merged/images" --gt_json_path "/data/datasets/hugging_face_td_dataset/open_tables_icttd_for_table_detection/Merged/merged_test.json" --config-file "configs/icttd_opentables.res50.300pro.yaml" --weight_path "/data/logs/icttd_opentable_merged_300/model_final.pth"

## Citing
Please cite our work if you think it is helpful:
```
@article{xiao2023table,
  title={Table detection for visually rich document images},
  author={Xiao, Bin and Simsek, Murat and Kantarci, Burak and Alkheir, Ala Abu},
  journal={Knowledge-Based Systems},
  volume={282},
  pages={111080},
  year={2023},
  publisher={Elsevier}
}
```
```
@article{xiao2023revisiting,
  title={Revisiting table detection datasets for visually rich documents},
  author={Xiao, Bin and Simsek, Murat and Kantarci, Burak and Alkheir, Ala Abu},
  journal={arXiv preprint arXiv:2305.04833},
  year={2023}
}
```
