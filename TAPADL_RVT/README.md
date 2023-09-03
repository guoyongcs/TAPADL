# Training and Evaluation of TAPADL based on RVT
[Robustifying Token Attention for Vision Transformers](https://arxiv.org/pdf/2303.11126.pdf), \
[Yong Guo](http://www.guoyongcs.com/), [David Stutz](https://davidstutz.de/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). ICCV 2023.



# Dependencies
Our code is built based on pytorch and timm library. Please check the detailed dependencies in [requirements.txt](https://github.com/guoyongcs/TAPADL/blob/main/requirements.txt).

# Dataset Preparation

Please download the clean [ImageNet](http://image-net.org/) dataset.


We use many robustness benchmarks for evaluation, including [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-C](https://zenodo.org/record/2235448), [ImageNet-P](https://zenodo.org/record/3565846) and [ImageNet-R](https://github.com/hendrycks/imagenet-r).


## Image Classification


### Pretrained Model

|       Model       | #Params | IN-1K $\uparrow$ | IN-C $\downarrow$ | IN-A $\uparrow$ | IN-P $\downarrow$ |
|:-----------------:|:----------------:|:-----------------:|:---------------:|:-----------------:|:-------:|
|  [RVT-B (TAP & ADL)](https://github.com/guoyongcs/TAPADL/releases/download/v1.0/tapadl_rvt_base.pth.tar)   |  92.1M  |     **83.1**     |     **44.7**      |    **32.7**     |     **29.6**      |

Please download and put the pretrained model [tapadl_rvt_base.pth.tar](https://github.com/guoyongcs/TAPADL/releases/download/v1.0/tapadl_rvt_base.pth.tar) in ```../pretrained```.


### Evaluation
- Evaluate the pretrained model on ImageNet:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 main.py --eval --model tap_rvt_base_plus --data-path /PATH/TO/IMAGENET --output_dir ./experiments/test_exp_tapadl_rvt_base_imagenet --dist-eval --pretrain_path ../pretrained/tapadl_rvt_base.pth.tar
```

- Evaluate the pretrained model on ImageNet-A/R:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 main.py --eval --model tap_rvt_base_plus --data-path /PATH/TO/IMAGENET --output_dir ./experiments/test_exp_tapadl_rvt_base_imagenet_a --dist-eval --pretrain_path ../pretrained/tapadl_rvt_base.pth.tar --ina_path /PATH/TO/IMAGENET-A
```

- Evaluate the pretrained model on ImageNet-C:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 main.py --eval --model tap_rvt_base_plus --data-path /PATH/TO/IMAGENET --output_dir ./experiments/test_exp_tapadl_rvt_base_imagenet_c --dist-eval --pretrain_path ../pretrained/tapadl_rvt_base.pth.tar --inc_path /PATH/TO/IMAGENET-C
```

- Evaluate the pretrained model on ImageNet-P

    Please refer to [test.sh](https://github.com/hendrycks/robustness/blob/master/ImageNet-P/test.sh) to see how to evaluate models on ImageNet-P.




### Training 
Train FAN-B-Hybrid with TAP and ADL on ImageNet (using 8 nodes and each with 8 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT  main.py --model tap_rvt_base_plus --data-path /PATH/TO/IMAGENET --output_dir ./experiments/exp_tapadl_rvt_base_imagenet --dist-eval --use_patch_aug --batch-size 64 --aa rand-m9-mstd0.5-inc1
```


## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{guo2023robustifying,
title={Robustifying token attention for vision transformers},
author={Guo, Yong and Stutz, David and Schiele, Bernt},
booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)}},
year={2023}
}
```





