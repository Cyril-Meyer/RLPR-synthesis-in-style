# RLPR - Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data #21
* [github issue](https://github.com/RLPR/LabelReviews/issues/21)
* [github repo](https://github.com/hendraet/synthesis-in-style)
* [paper](https://arxiv.org/pdf/2107.06777.pdf)

###  Results that can be reproduced
#### Tables I - VI
Tables I - III are short version of tables IV - V.

in progress
![image](https://user-images.githubusercontent.com/69190238/169360129-53c2d45b-93c6-467f-9e85-4af182568250.png)

#### Figures 3 - 7
Figure 3 = Figure 4

| fig 4 | fig 5 | fig 6 | fig 7 |
| ----- | ----- | ----- | ----- |
| ![figures 4](figures/89r2yz1e4hd90_both_min_confidence_0_9_min_contour_area_15_patch_overlap_0__0_5_segmented_no_bboxes.png) | ![figures 5](figures/ozey203xgb369_printed_over_image_min_confidence_0_9_min_contour_area_15_patch_overlap_0__0_5_segmented_no_bboxes.png) | ![figures 6](figures/kynugqpw6mmsj_cover_greenish_min_confidence_0_9_min_contour_area_15_patch_overlap_0__0_5_segmented_no_bboxes.png) | ![figures 7](figures/WPI_GACHET_2018_010_0160_handwritten_printed_greyish_min_confidence_0_9_min_contour_area_15_patch_overlap_0__0_5_segmented_no_bboxes.png)


### Get the code and data
```
git clone git@github.com:hendraet/synthesis-in-style.git
```

The trained models and benchmark dataset used for evaluation can be found here : [https://bartzi.de/research/synthesis_in_style](https://bartzi.de/research/synthesis_in_style)
```
wget https://bartzi.de/documents/attachment/download\?hash_value\=930173d8e073227e1845b4dea8abfc0c_57 -O benchmark_dataset.zip
wget https://bartzi.de/documents/attachment/download\?hash_value\=dd15bbce1b42200ed347ecc3ab98993b_66 -O sis_swagan.tar.gz
wget https://bartzi.de/documents/attachment/download\?hash_value\=9ccec0a8c423832554a2c4babc9679d4_67 -O sis_stylegan.tar.gz
wget https://bartzi.de/documents/attachment/download?hash_value=1c52f3da5539c3b2be134f2f6d6753d2_68 -O datasetgan_swagan.tar.gz
wget https://bartzi.de/documents/attachment/download?hash_value=b6d671c7fb5e625cacbc10509789498c_69 -O datasetgan_stylegan.tar.gz
```

```
unzip benchmark_dataset.zip
7z x -so sis_swagan.tar.gz | 7z x -si -ttar
7z x -so sis_stylegan.tar.gz | 7z x -si -ttar
7z x -so datasetgan_swagan.tar.gz | 7z x -si -ttar
7z x -so datasetgan_stylegan.tar.gz | 7z x -si -ttar
```

### Using the provided docker image

```
# pull docker
docker pull hendraet/synthesis-in-style:cuda-11.1
# test docker work with gpu
docker run -it --rm --gpus all hendraet/synthesis-in-style:cuda-11.1
python3.8 # python = 2.7.18, python3.8 = 3.8.10
```

```
>>> import torch
>>> torch.version.cuda
'11.1'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
2
>>> torch.cuda.current_device()
0
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 2080 Ti'
```

### Evaluate provided segmentation models
Create config file

*trans_u_net*
```
vim synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_config.json
```
```
{
  "checkpoint": "/rlpr/sis_stylegan/trans_u_net.pt",
  "class_to_color_map": "handwriting_colors.json",
  "max_image_size": 0
}
```
```
vim synthesis-in-style/stylegan_code_finder/configs/evaluation/datasetgan_stylegan_config.json
```
```
{
  "checkpoint": "/rlpr/datasetgan_stylegan/trans_u_net.pt",
  "class_to_color_map": "handwriting_colors.json",
  "max_image_size": 0
}
```

*Doc-UFCN*
```
vim synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_ufcn_config.json
```
```
{
  "checkpoint": "/rlpr/sis_stylegan/doc_ufcn.pt",
  "class_to_color_map": "handwriting_colors.json",
  "max_image_size": 0
}
```
```
vim synthesis-in-style/stylegan_code_finder/configs/evaluation/datasetgan_stylegan_ufcn_config.json
```
```
{
  "checkpoint": "/rlpr/datasetgan_stylegan/doc_ufcn.pt",
  "class_to_color_map": "handwriting_colors.json",
  "max_image_size": 0
}
```

*EMANet*
```
vim synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_emanet_config.json
```
```
{
  "checkpoint": "/rlpr/sis_stylegan/ema_net.pt",
  "class_to_color_map": "handwriting_colors.json",
  "max_image_size": 0
}
```
```
vim synthesis-in-style/stylegan_code_finder/configs/evaluation/datasetgan_stylegan_emanet_config.json
```
```
{
  "checkpoint": "/rlpr/datasetgan_stylegan/ema_net.pt",
  "class_to_color_map": "handwriting_colors.json",
  "max_image_size": 0
}
```

Create result folder
```
# replace "/home/cyril/Development/RLPR" with "$(pwd)"
docker run -v /home/cyril/Development/RLPR:/rlpr -it --rm --gpus all hendraet/synthesis-in-style:cuda-11.1
```

```
cd /rlpr/synthesis-in-style/stylegan_code_finder/ 
```

*trans_u_net*

```
PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_config.json \
  --original-config-path /rlpr/synthesis-in-style/stylegan_code_finder/configs/segmenter/stylegan2_trans_u_net_segmenter.yaml \
  --output-dir /rlpr/out_sis_stylegan \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis
```

```
PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/datasetgan_stylegan_config.json \
  --original-config-path /rlpr/synthesis-in-style/stylegan_code_finder/configs/segmenter/stylegan2_trans_u_net_segmenter.yaml \
  --output-dir /rlpr/out_datasetgan_stylegan \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis
```

*Doc-UFCN*
```
PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_ufcn_config.json \
  --original-config-path /rlpr/synthesis-in-style/stylegan_code_finder/configs/segmenter/stylegan2_doc_ufcn_segmenter.yaml \
  --output-dir /rlpr/out_sis_stylegan_ufcn \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis

PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/datasetgan_stylegan_ufcn_config.json \
  --original-config-path /rlpr/synthesis-in-style/stylegan_code_finder/configs/segmenter/stylegan2_doc_ufcn_segmenter.yaml \
  --output-dir /rlpr/out_datasetgan_stylegan_ufcn \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis
```

*EMANet*
```
PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_emanet_config.json \
  --original-config-path /rlpr/synthesis-in-style/stylegan_code_finder/configs/segmenter/stylegan2_ema_net_segmenter.yaml \
  --output-dir /rlpr/out_sis_stylegan_emanet \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis

PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/datasetgan_stylegan_emanet_config.json \
  --original-config-path /rlpr/synthesis-in-style/stylegan_code_finder/configs/segmenter/stylegan2_ema_net_segmenter.yaml \
  --output-dir /rlpr/out_datasetgan_stylegan_emanet \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis
```
 
Extract result and print metrics
```
# in-domain - ours - TransUNet
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_sis_stylegan/results.json /rlpr/benchmark_dataset/in-domain-images.txt /rlpr/out_sis_stylegan/results_in-domain-images.json
# check post-processing hyperparameters values
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan/results_in-domain-images.json -c -p
# check score values
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan/results_in-domain-images.json -c -r min_confidence 0.9 min_contour_area 15 patch_overlap 0.5
```
|    |   min_confidence |   min_contour_area |   patch_overlap |   dice_weighted_avg |   dice_weighted_text_avg |   dice_background |   dice_printed_text |   dice_handwritten_text |   iou_weighted_avg |   iou_weighted_text_avg |   iou_background |   iou_printed_text |   iou_handwritten_text |   precision_weighted_avg |   precision_weighted_text_avg |   precision_background |   precision_printed_text |   precision_handwritten_text |   recall_weighted_avg |   recall_weighted_text_avg |   recall_background |   recall_printed_text |   recall_handwritten_text |
|---:|-----------------:|-------------------:|----------------:|--------------------:|-------------------------:|------------------:|--------------------:|------------------------:|-------------------:|------------------------:|-----------------:|-------------------:|-----------------------:|-------------------------:|------------------------------:|-----------------------:|-------------------------:|-----------------------------:|----------------------:|---------------------------:|--------------------:|----------------------:|--------------------------:|
|  0 |              0.9 |                 15 |             0.5 |            0.996268 |                 0.821749 |          0.998238 |            0.833186 |                0.632954 |            0.72452 |                0.699729 |         0.996482 |           0.714069 |               0.463008 |                 0.996241 |                      0.833212 |                0.99808 |                 0.842972 |                     0.672105 |              0.996301 |                   0.810744 |            0.998395 |              0.823625 |                  0.598112 |

```
# out-domain - ours - TransUNet
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_sis_stylegan/results.json /rlpr/benchmark_dataset/out-of-domain-images.txt /rlpr/out_sis_stylegan/results_out-domain-images.json
# not same post process hyperparam
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan/results_out-domain-images.json -c -p
# force post process hyperparam
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan/results_out-domain-images.json -c -r min_confidence 0.9 min_contour_area 15 patch_overlap 0.5
```
|    |   min_confidence |   min_contour_area |   patch_overlap |   dice_weighted_avg |   dice_weighted_text_avg |   dice_background |   dice_printed_text |   dice_handwritten_text |   iou_weighted_avg |   iou_weighted_text_avg |   iou_background |   iou_printed_text |   iou_handwritten_text |   precision_weighted_avg |   precision_weighted_text_avg |   precision_background |   precision_printed_text |   precision_handwritten_text |   recall_weighted_avg |   recall_weighted_text_avg |   recall_background |   recall_printed_text |   recall_handwritten_text |
|---:|-----------------:|-------------------:|----------------:|--------------------:|-------------------------:|------------------:|--------------------:|------------------------:|-------------------:|------------------------:|-----------------:|-------------------:|-----------------------:|-------------------------:|------------------------------:|-----------------------:|-------------------------:|-----------------------------:|----------------------:|---------------------------:|--------------------:|----------------------:|--------------------------:|
|  0 |              0.9 |                 15 |             0.5 |            0.981201 |                 0.573953 |          0.992346 |            0.635311 |                0.294833 |           0.541083 |                  0.4128 |         0.984809 |           0.465536 |               0.172905 |                 0.981267 |                      0.778647 |               0.986812 |                 0.865515 |                     0.383484 |              0.983468 |                   0.454556 |            0.997943 |              0.501837 |                  0.239473 |

```
# in-domain - DatasetGAN - TransUNet
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_datasetgan_stylegan/results.json /rlpr/benchmark_dataset/in-domain-images.txt /rlpr/out_datasetgan_stylegan/results_in-domain-images.json
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan/results_in-domain-images.json -c -p
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan/results_in-domain-images.json -c -r min_confidence 0.7 min_contour_area 55 patch_overlap 0.5
```
|    |   min_confidence |   min_contour_area |   patch_overlap |   dice_weighted_avg |   dice_weighted_text_avg |   dice_background |   dice_printed_text |   dice_handwritten_text |   iou_weighted_avg |   iou_weighted_text_avg |   iou_background |   iou_printed_text |   iou_handwritten_text |   precision_weighted_avg |   precision_weighted_text_avg |   precision_background |   precision_printed_text |   precision_handwritten_text |   recall_weighted_avg |   recall_weighted_text_avg |   recall_background |   recall_printed_text |   recall_handwritten_text |
|---:|-----------------:|-------------------:|----------------:|--------------------:|-------------------------:|------------------:|--------------------:|------------------------:|-------------------:|------------------------:|-----------------:|-------------------:|-----------------------:|-------------------------:|------------------------------:|-----------------------:|-------------------------:|-----------------------------:|----------------------:|---------------------------:|--------------------:|----------------------:|--------------------------:|
|  0 |              0.7 |                 55 |             0.5 |            0.994908 |                 0.735855 |          0.997831 |            0.757059 |                0.385834 |           0.614596 |                 0.58795 |         0.995671 |           0.609087 |                0.23903 |                  0.99506 |                      0.800846 |               0.997252 |                 0.831562 |                     0.293815 |              0.994938 |                    0.68721 |             0.99841 |              0.694808 |                  0.561774 |

```
# out-domain - DatasetGAN - TransUNet
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_datasetgan_stylegan/results.json /rlpr/benchmark_dataset/out-of-domain-images.txt /rlpr/out_datasetgan_stylegan/results_out-domain-images.json
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan/results_out-domain-images.json -c -p
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan/results_out-domain-images.json -c -r min_confidence 0.7 min_contour_area 55 patch_overlap 0.5
```
|    |   min_confidence |   min_contour_area |   patch_overlap |   dice_weighted_avg |   dice_weighted_text_avg |   dice_background |   dice_printed_text |   dice_handwritten_text |   iou_weighted_avg |   iou_weighted_text_avg |   iou_background |   iou_printed_text |   iou_handwritten_text |   precision_weighted_avg |   precision_weighted_text_avg |   precision_background |   precision_printed_text |   precision_handwritten_text |   recall_weighted_avg |   recall_weighted_text_avg |   recall_background |   recall_printed_text |   recall_handwritten_text |
|---:|-----------------:|-------------------:|----------------:|--------------------:|-------------------------:|------------------:|--------------------:|------------------------:|-------------------:|------------------------:|-----------------:|-------------------:|-----------------------:|-------------------------:|------------------------------:|-----------------------:|-------------------------:|-----------------------------:|----------------------:|---------------------------:|--------------------:|----------------------:|--------------------------:|
|  0 |              0.7 |                 55 |             0.5 |            0.977312 |                 0.452578 |          0.991673 |            0.503234 |                0.222141 |           0.481549 |                0.298142 |         0.983483 |           0.336214 |               0.124948 |                 0.978855 |                      0.748133 |               0.985169 |                 0.863337 |                     0.224066 |              0.980483 |                    0.33081 |            0.998263 |              0.355114 |                  0.220248 |

```
# in-domain - ours - DocUFCN
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_sis_stylegan_ufcn/results.json /rlpr/benchmark_dataset/in-domain-images.txt /rlpr/out_sis_stylegan_ufcn/results_in-domain-images.json
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan_ufcn/results_in-domain-images.json -c -p
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan_ufcn/results_in-domain-images.json -c -r min_confidence 0.3 min_contour_area 15 patch_overlap 0.5

# out-domain - ours - DocUFCN
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_sis_stylegan_ufcn/results.json /rlpr/benchmark_dataset/out-of-domain-images.txt /rlpr/out_sis_stylegan_ufcn/results_out-domain-images.json
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_sis_stylegan_ufcn/results_out-domain-images.json -c -r min_confidence 0.3 min_contour_area 15 patch_overlap 0.5

# in-domain - DatasetGAN - DocUFCN
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_datasetgan_stylegan_ufcn/results.json /rlpr/benchmark_dataset/in-domain-images.txt /rlpr/out_datasetgan_stylegan_ufcn/results_in-domain-images.json
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan_ufcn/results_in-domain-images.json -c -p
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan_ufcn/results_in-domain-images.json -c -r min_confidence 0.3 min_contour_area 15 patch_overlap 0

# out-domain - DatasetGAN - DocUFCN
PYTHONPATH='.' python3.8 ./segmentation/evaluation/extract_results_subset.py /rlpr/out_datasetgan_stylegan_ufcn/results.json /rlpr/benchmark_dataset/out-of-domain-images.txt /rlpr/out_datasetgan_stylegan_ufcn/results_out-domain-images.json
PYTHONPATH='.' python3.8 ./segmentation/evaluation/evaluate_metrics.py /rlpr/out_datasetgan_stylegan_ufcn/results_out-domain-images.json -c -r min_confidence 0.3 min_contour_area 15 patch_overlap 0
```
|    |   min_confidence |   min_contour_area |   patch_overlap |   dice_weighted_avg |   dice_weighted_text_avg |   dice_background |   dice_printed_text |   dice_handwritten_text |   iou_weighted_avg |   iou_weighted_text_avg |   iou_background |   iou_printed_text |   iou_handwritten_text |   precision_weighted_avg |   precision_weighted_text_avg |   precision_background |   precision_printed_text |   precision_handwritten_text |   recall_weighted_avg |   recall_weighted_text_avg |   recall_background |   recall_printed_text |   recall_handwritten_text |
|---:|-----------------:|-------------------:|----------------:|--------------------:|-------------------------:|------------------:|--------------------:|------------------------:|-------------------:|------------------------:|-----------------:|-------------------:|-----------------------:|-------------------------:|------------------------------:|-----------------------:|-------------------------:|-----------------------------:|----------------------:|---------------------------:|--------------------:|----------------------:|--------------------------:|
|  0 |              0.3 |                 15 |             0.5 |            0.995627 |                 0.799051 |          0.997845 |            0.820852 |                0.439182 |           0.657739 |                0.672449 |         0.995699 |           0.696139 |               0.281379 |                  0.99592 |                      0.796339 |               0.998172 |                 0.825199 |                     0.319947 |              0.995425 |                   0.809898 |            0.997518 |               0.81655 |                  0.700081 |
|  0 |              0.3 |                 15 |             0.5 |            0.977673 |                 0.481646 |          0.991248 |            0.545311 |                0.192034 |           0.487909 |                0.326451 |         0.982647 |           0.374864 |               0.106215 |                 0.980357 |                      0.744998 |               0.986798 |                 0.874891 |                     0.154115 |              0.979086 |                   0.370616 |            0.995738 |              0.396097 |                  0.254702 |
|  0 |              0.3 |                 15 |               0 |            0.995054 |                 0.766436 |          0.997634 |            0.783077 |                0.491728 |            0.65493 |                0.625356 |         0.995279 |            0.64349 |                0.32602 |                 0.994961 |                      0.815628 |               0.996984 |                  0.82723 |                     0.624111 |              0.995225 |                   0.724109 |            0.998284 |              0.743399 |                  0.405677 |
|  0 |              0.3 |                 15 |               0 |            0.974935 |                 0.421994 |          0.990067 |            0.490402 |                0.110808 |           0.454613 |                0.276883 |          0.98033 |           0.324856 |              0.0586538 |                 0.974795 |                      0.704286 |               0.982198 |                  0.80184 |                     0.260512 |              0.979528 |                   0.302241 |            0.998063 |              0.353212 |                   0.07037 |

 
### Missing details
Using the provided docker :
* the default python is 2.7, you need to use python3 or python3.8
* you need to add PYTHONPATH='.' before python call to avoid import errors

**negligible errors**
* Table V | handwritten text | recall - ours - TransUNet - rounding error for 0.239473
