# RLPR - Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data #21
* [github issue](https://github.com/RLPR/LabelReviews/issues/21)
* [github repo](https://github.com/hendraet/synthesis-in-style)
* [paper](https://arxiv.org/pdf/2107.06777.pdf)

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
# replace "/home/cyril/Development/RLPR" with "$(pwd)"
docker run -v /home/cyril/Development/RLPR:/rlpr -it --rm --gpus all hendraet/synthesis-in-style:cuda-11.1
```

```
cd /rlpr/synthesis-in-style/stylegan_code_finder/ 

PYTHONPATH='.' python3.8 ./segmentation/evaluation/analyze_image_segments.py \
  /rlpr/benchmark_dataset/original \
  -gt /rlpr/benchmark_dataset/ground_truth \
  --config-file /rlpr/synthesis-in-style/stylegan_code_finder/configs/evaluation/sis_stylegan_config.json \
  --output-dir out \
  -bw \
  --patch-overlap-factor 0.50 0.0 \
  --min-confidence 0.3 0.7 0.9 \
  --min-contour-area 15 30 55 \
  -cds -cre -cpr -cio \
  -vis
 ```


### Missing details
Using the provided docker :
* the default python is 2.7, you need to use python3 or python3.8
* you need to add PYTHONPATH='.' before python call to avoid import errors

