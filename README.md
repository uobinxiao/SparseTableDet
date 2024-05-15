# Table detection for visually rich document images

## Paper Link
https://www.sciencedirect.com/science/article/abs/pii/S0950705123008304 \
https://arxiv.org/abs/2305.19181

## Dataset
You can download the table detection dataset with the following link:
```
https://huggingface.co/datasets/uobinxiao/open_tables_icttd_for_table_detection
```
## Pretrained Model
A pre-trained model can be downloaded [here](https://drive.google.com/drive/folders/1dQUVgqI0894EaNo7WFecrw7BA2ZlDevD?usp=sharing), which is trained by the [OpenTables&ICT-TD dataset](https://huggingface.co/datasets/uobinxiao/open_tables_icttd_for_table_detection). It is worth mentioning that the merged training sets and the merged testing sets of OpenTables and ICT-TD are used for training and evaluation. The evaluation scores are as following:

| MAP | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 | 
| --- |  --- | --- |   --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- | 
|0.954| 0.980|0.978| 0.977| 0.977| 0.973| 0.969| 0.964| 0.954| 0.933| 0.837| 

| MAR | AR50 | AR55 | AR60 | AR65 | AR70 | AR75 | AR80 | AR85 | AR90 | AR95 |
| --- |  --- | --- |   --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- | 
| 0.980| 0.999| 0.998| 0.998| 0.997| 0.995| 0.994| 0.991| 0.985| 0.966| 0.881|
## Requirements
This codebase is built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [Sparse-RCNN](https://github.com/PeizeSun/SparseR-CNN/tree/main).
Follow the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install Detectron2.

## Configuration and Training
Set config.yaml based on the dataset.\
Set configs/icttd_opentables.res50.300pro.yaml to modify the model parameters and the output log directory.\
Use python train_net.py to train the model.

## Evaluation
Use python predict.py to evaluate the model.

```
python predict.py --input_dir <image_dir> --gt_json_path <path of the ground truth json file> --config-file <path of the config yaml file> --weight_path <path of the weight file>
```

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
