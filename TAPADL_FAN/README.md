# Training and evaluation of TAPADL based on FAN for image classification and semantic segmentation
[Robustifying Token Attention for Vision Transformers](https://arxiv.org/pdf/2303.11126.pdf), \
[Yong Guo](http://www.guoyongcs.com/), [David Stutz](https://davidstutz.de/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). ICCV 2023.



# Dependencies
Our code is built based on pytorch and timm library. Please check the detailed dependencies in [requirements.txt](https://github.com/guoyongcs/TAPADL/blob/main/requirements.txt).

# Dataset Preparation

Please download the clean [ImageNet](http://image-net.org/) dataset and [ImageNet-C](https://zenodo.org/record/2235448) dataset and structure the datasets as follows:

```
/PATH/TO/IMAGENET-C/
  clean/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
  corruption1/
    severity1/
      class1/
        img3.jpeg
      class2/
        img4.jpeg
    severity2/
      class1/
        img3.jpeg
      class2/
        img4.jpeg
```

We also use other robustness benchmarks for evaluation, including [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-P](https://zenodo.org/record/3565846) and [ImageNet-R](https://github.com/hendrycks/imagenet-r).



## Image Classification



### Pretrained Model

|       Model       | #Params | IN-1K $\uparrow$ | IN-C $\downarrow$ | IN-A $\uparrow$ | IN-P $\downarrow$ |
|:-----------------:|:----------------:|:-----------------:|:---------------:|:-----------------:|:-------:|
|  [FAN-B-ViT (TAP & ADL)](https://github.com/guoyongcs/TAPADL/releases/download/v1.0/tapadl_fan_base.pth.tar)   |  50.7M  |     **84.3**     |     **43.7**      |    **42.3**     |     **29.2**      |

Please download and put the pretrained model [tapadl_fan_base.pth.tar](https://github.com/guoyongcs/TAPADL/releases/download/v1.0/tapadl_fan_base.pth.tar) in ```../pretrained```.


### Evaluation
- Evaluate the pretrained model on ImageNet:
```
CUDA_VISIBLE_DEVICES=0 python validate_ood.py /PATH/TO/IMAGENET --model tap_fan_base_16_p4_hybrid \
    --checkpoint ../pretrained/tapadl_fan_base.pth.tar --num-gpu 1 --amp --num-scales 4
```

- Evaluate the pretrained model on ImageNet-A/R:
```
CUDA_VISIBLE_DEVICES=0 python validate_ood.py /PATH/TO/IMAGENET-A --model tap_fan_base_16_p4_hybrid \
    --checkpoint ../pretrained/tapadl_fan_base.pth.tar --num-gpu 1 --amp --num-scales 4 --imagenet_a
```

- Evaluate the pretrained model on ImageNet-C:
```
CUDA_VISIBLE_DEVICES=0 python validate_ood.py /PATH/TO/IMAGENET --model tap_fan_base_16_p4_hybrid \
    --checkpoint ../pretrained/tapadl_fan_base.pth.tar --num-gpu 1 --imagenet_c \
    --inc_path /PATH/TO/IMAGENET-C --amp --num-scales 4
```

- Evaluate the pretrained model on ImageNet-P

    Please refer to [test.sh](https://github.com/hendrycks/robustness/blob/master/ImageNet-P/test.sh) to see how to evaluate models on ImageNet-P.




### Training 
Train FAN-B-Hybrid with TAP and ADL on ImageNet (using 8 nodes and each with 8 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=$NODE_RANK \ 
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py /PATH/TO/IMAGENET \ 
    --model tap_fan_base_16_p4_hybrid -b 64 --sched cosine --opt adamw -j 16 --warmup-lr 1e-6 \
    --warmup-epochs 10 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.3 --lr 40e-4 \
    --min-lr 1e-6 --weight-decay .05 --drop 0.0 --drop-path .35 --img-size 224 --mixup 0.8 \
    --cutmix 1.0 --smoothing 0.1 --output ./experiments/exp_tapadl_fan_base_imagenet \
    --amp --model-ema
```



## Semantic Segmentation

Please see details in [README.md](https://github.com/guoyongcs/TAPADL/blob/main/TAPADL_FAN/segmentation).


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





