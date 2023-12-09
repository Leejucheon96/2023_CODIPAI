CUDA_VISIBLE_DEVICES=7 python train_end2end.py --exp_code 'FT_codipai_res50_512' \
--k_start 0 --k_end 5 \
--bag_size 512 \
--data_root_dir '/data3/CMCS/raw_data_20x/Codipai_patch256_ostu/topk_patches_512/' \