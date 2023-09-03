import os
import sys

# 2768683-2768693
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0826_imnet_rvt_tiny_plus_nt1_aw3_gaussiannoise'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-noisy-tokens 1 --aux-weight 3 --tokenaug-type gaussian_noise --use_patch_aug'

# 2768696
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0826_imnet_rvt_tiny_plus_noaa_nt1_aw3_gaussiannoise'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 3 --tokenaug-type gaussian_noise --use_patch_aug'

# 2768710
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0826_imnet_rvt_tiny_plus_nt1_aw3_gaussiannoise_nopatchaug'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-noisy-tokens 1 --aux-weight 3 --tokenaug-type gaussian_noise'


#0828
# 2771250-2771267
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_baseline'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug'

# 2771404
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_baseline'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --use_patch_aug'

# 2771416
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_nt1_aw1_gaussiannoise'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 1 --tokenaug-type gaussian_noise --use_patch_aug'

# 2771426
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_nt1_aw05_gaussiannoise'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 0.5 --tokenaug-type gaussian_noise --use_patch_aug'

# 2771437
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_nt1_aw01_gaussiannoise'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 0.1 --tokenaug-type gaussian_noise --use_patch_aug'


# 2773173
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_nt1_aw05_gaussiannoise_nopatchaug'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 0.5 --tokenaug-type gaussian_noise'

# 2773189
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_nt1_aw01_gaussiannoise_nopatchaug'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 0.1 --tokenaug-type gaussian_noise'

# 2773399
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0828_imnet_rvt_tiny_plus_noaa_baseline_nopatchaug'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128'



# 2786063
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0831_imnet_rvt_tiny_plus_noaa_nopatchaug_aw0'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 0 --tokenaug-type gaussian_noise --aux-loss-type ce --inc_path /ptmp/yguo/imagenet-c'

#
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0831_imnet_rvt_tiny_plus_noaa_nopatchaug_aw1_ce'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --num-noisy-tokens 1 --aux-weight 1 --tokenaug-type gaussian_noise --aux-loss-type ce --inc_path /ptmp/yguo/imagenet-c'

# 2812852
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0903_imnet_rvt_tiny_plus_noaa_nopatchaug_nt_aw04'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0.4 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c'

# 2812862
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0903_imnet_rvt_tiny_plus_noaa_nopatchaug_nt_aw01'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0.1 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c'

# 2812883
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0903_imnet_rvt_tiny_plus_noaa_baseline_nopatchaug'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --inc_path /ptmp/yguo/imagenet-c'


# 2817827
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_nt_aw0'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c'

# 2817837
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_nt_aw001'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model nt_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0.01 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c'


# 2818241
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_ntr_aw03'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model ntr_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0.3 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c'


# 2818253
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_ntr_aw01'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model ntr_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0.1 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c'

# 2818263
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_ntr_aw0_nl0'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model ntr_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128  --aux-weight 0 --num-noisy-tokens 1 --inc_path /ptmp/yguo/imagenet-c --noise-level 0'


# 2820055
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_ms'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model ms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --inc_path /ptmp/yguo/imagenet-c'


# 2820067
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0904_imnet_rvt_tiny_plus_noaa_nopatchaug_lms'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model lms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --aa None --batch-size 128 --inc_path /ptmp/yguo/imagenet-c'




# 2858618
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0908_imnet_rvt_tiny_plus_lms_ns3_t1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --num-scales 3 --temperature 1'




#2865637
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0909_imnet_rvt_tiny_plus_lms_ns3_t1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model lms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --num-scales 3 --temperature 1'

#2865648
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0909_imnet_rvt_tiny_plus_llms_ns3_t1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model llms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --num-scales 3 --temperature 1'

# 2866792
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0909_imnet_rvt_tiny_plus_llms_ns5_t1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model llms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --num-scales 5 --temperature 1'

# 2866803
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0909_imnet_rvt_tiny_plus_llms_ns7_t1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model llms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --num-scales 7 --temperature 1'


# 2877342
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0911_imnet_rvt_tiny_plus_lms_ns3_t1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model lms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --num-scales 3 --temperature 1 --inc_path /ptmp/yguo/imagenet-c'


# 2877353
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0911_imnet_rvt_tiny_plus_baseline'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --use_patch_aug --inc_path /ptmp/yguo/imagenet-c'


# 2877935
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0911_imnet_rvt_tiny_plus_lms_ns3_t1_nobias_nopa'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model lms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-scales 3 --temperature 1 --inc_path /ptmp/yguo/imagenet-c'


# 2877951
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0911_imnet_rvt_tiny_plus_baseline_nopa'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --inc_path /ptmp/yguo/imagenet-c'

# 2878288
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0911_imnet_rvt_tiny_plus_llms_ns3_t1_nobias_nopa'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model llms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-scales 3 --temperature 1 --inc_path /ptmp/yguo/imagenet-c'


#  2878298
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/0911_imnet_rvt_tiny_plus_llms_ns5_t1_nobias_nopa'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model llms_rvt_tiny_plus --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-scales 5 --temperature 1 --inc_path /ptmp/yguo/imagenet-c'

# 3420434 3507689
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20221126_imnet_rvt_tiny_plus_mdh_ns3'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus_mdh --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-scales 3 --inc_path /ptmp/yguo/imagenet-c --use_patch_aug --model-ema-decay 0.9998'

# 3420440 3507692
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20221126_imnet_rvt_tiny_plus_mdh_ns3_epoch350'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus_mdh --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --batch-size 128 --num-scales 3 --inc_path /ptmp/yguo/imagenet-c --use_patch_aug --epochs 350 --model-ema-decay 0.9998'




# 3940506 4110458
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20230119_imnet_rvt_base_plus_exmd_ns4_tl1'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_base_plus_exmd --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --use_patch_aug --epochs 300 --batch-size 64 --aa rand-m9-mstd0.5-inc1 --inc_path /ptmp/yguo/imagenet-c --model-ema-decay 0.9998 --deepaugment --deepaugment_base_path /ptmp/yguo/DeepAugment --deepaug_freq 150 --threshold 2 --num-scales 4 --token-loss-weight 1'


# 3940510 4110466
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20230119_imnet_rvt_base_plus_exmd_ns4_tl2'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_base_plus_exmd --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --use_patch_aug --epochs 300 --batch-size 64 --aa rand-m9-mstd0.5-inc1 --inc_path /ptmp/yguo/imagenet-c --model-ema-decay 0.9998 --deepaugment --deepaugment_base_path /ptmp/yguo/DeepAugment --deepaug_freq 150 --threshold 2 --num-scales 4 --token-loss-weight 2'

# 3940514 4110470
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20230119_imnet_rvt_base_plus_exmd_ns4_tl1_epoch350'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_base_plus_exmd --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --use_patch_aug --epochs 350 --batch-size 64 --aa rand-m9-mstd0.5-inc1 --inc_path /ptmp/yguo/imagenet-c --model-ema-decay 0.9998 --deepaugment --deepaugment_base_path /ptmp/yguo/DeepAugment --deepaug_freq 150 --threshold 2 --num-scales 4 --token-loss-weight 1'



# 0207 ft

# 4178093 4183554
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20230210_imnet_rvt_base_plus_exmd_ns4_tl1_ft'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_base_plus_exmd --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --use_patch_aug --epochs 70 --batch-size 64 --aa rand-m9-mstd0.5-inc1 --inc_path /ptmp/yguo/imagenet-c --deepaugment --deepaugment_base_path /ptmp/yguo/DeepAugment --deepaug_freq 150 --threshold 2 --num-scales 4 --token-loss-weight 1 --eval_model_ema --pretrain_path /u/yguo/mycode/ADViT/experiments/20230119_imnet_rvt_base_plus_exmd_ns4_tl1/model_best.pth.tar --lr 1e-5 --min-lr 5e-5'



# 4178096 4183556
# MARKER_FILE='/u/yguo/mycode/ADViT/experiments/20230210_imnet_rvt_base_plus_exmd_ns4_tl1_epoch350_ft'
# a = 'python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_base_plus_exmd --data-path /u/yguo/dataset/ILSVRC2012_copy --output_dir $MARKER_FILE --dist-eval --use_patch_aug --epochs 70 --batch-size 64 --aa rand-m9-mstd0.5-inc1 --inc_path /ptmp/yguo/imagenet-c --deepaugment --deepaugment_base_path /ptmp/yguo/DeepAugment --deepaug_freq 150 --threshold 2 --num-scales 4 --token-loss-weight 1 --eval_model_ema --pretrain_path /u/yguo/mycode/ADViT/experiments/20230119_imnet_rvt_base_plus_exmd_ns4_tl1_epoch350/model_best.pth.tar --lr 1e-5 --min-lr 5e-5'





a = a.replace('$NODE_RANK', str(os.environ['SLURM_NODEID']))
a = a.replace('$MASTER_ADDR', str(os.environ['MASTER_ADDR']))
a = a.replace('$MASTER_PORT', str(os.environ['MASTER_PORT']))
a = a.replace('$MARKER_FILE', MARKER_FILE)
print(a)
os.system(a)
