CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py --source ./data_wsis_20x/ \
--save_dir ./data_wsis_20x/Codipai_patch256_ostu \
--patch_level 0 \
--patch_size 256 \
--step_size 256 --seg --use_ostu --patch