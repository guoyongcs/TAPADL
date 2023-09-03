# Segmentation codebase for TAP and ADL based on FAN

We follow [FAN](https://github.com/NVlabs/FAN/tree/master) to build our codebase which is developed on top of [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).


## Dependencies

Install according to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Please refer to [requirements.txt](https://github.com/guoyongcs/TAPADL/blob/main/requirements.txt) for other dependencies.


## Dataset Preparation

- Prepare Cityscapes according to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).


- To generate Cityscapes-C dataset, first install the natural image corruption lib via:

pip install imagecorruptions

Then, run the following command:

```
python tools/gen_city_c.py
```

Please see more details of generating Cityscapes-C in the [guidelines](https://github.com/NVlabs/FAN/tree/master/segmentation).


- As for ACDC, please refer to [acdc.vision.ee.ethz.ch](https://acdc.vision.ee.ethz.ch) to get the test set and submit the results.

## Pretrained Model

Please put the pretrained model [tapadl_fan_base_segmentation.pth](tapadl_fan_base_segmentation.pth) in ```../../pretrained```.


## Evaluation

- Evaluate the pretrained model on Cityscapes:

1. Please specify the data path "data_root" in [cityscapes_1024x1024_repeat.py](https://github.com/guoyongcs/TAPADL/blob/main/TAPADL_FAN/segmentation/local_configs/_base_/datasets/cityscapes_1024x1024_repeat.py).


2. Evaluate the model via
```
CUDA_VISIBLE_DEVICES=0 python test_cityscapes.py local_configs/fan/fan_hybrid/tapfan_hybrid_base.1024x1024.city.160k.test.py ../../pretrained/tapadl_fan_base_segmentation.pth --eval mIoU --results-file output/
```



- Evaluate the pretrained model on Cityscapes-C:

1. Please specify the data path "data_root" in [cityscapes_1024x1024_repeat_cityc.py](https://github.com/guoyongcs/TAPADL/blob/main/TAPADL_FAN/segmentation/local_configs/_base_/datasets/cityscapes_1024x1024_repeat_cityc.py).


2. Evaluate the model via
```
CUDA_VISIBLE_DEVICES=0 python test_cityscapes_c.py local_configs/fan/fan_hybrid/tapfan_hybrid_base.1024x1024.city.160k.test.py ../../pretrained/tapadl_fan_base_segmentation.pth --eval mIoU --results-file output/
```


- Evaluate the pretrained model on ACDC (test set):

1. Please specify the data path "data_root" in [cityscapes_1024x1024_repeat_acdc.py](https://github.com/guoyongcs/TAPADL/blob/main/TAPADL_FAN/segmentation/local_configs/_base_/datasets/cityscapes_1024x1024_repeat_acdc.py).


2. Evaluate the model via
```
CUDA_VISIBLE_DEVICES=0 python test_cityscapes_c.py local_configs/fan/fan_hybrid/tapfan_hybrid_base.1024x1024.city.160k.test.py ../../pretrained/tapadl_fan_base_segmentation.pth --results-file output/ --show-dir output/
```

3. Submit the results to obtain the mIoU score on [acdc.vision.ee.ethz.ch](https://acdc.vision.ee.ethz.ch)



## Training

Train FAN-B-Hybrid with TAP and ADL on Cityscapes (using 4 GPUs)

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py local_configs/fan/fan_hybrid/tapadl_fan_hybrid_base.1024x1024.city.160k.py --launcher pytorch --work-dir ./exp_tapadl_fan_base_segmentation_cityscapes --auto-resume
```

