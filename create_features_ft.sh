CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
--data_h5_dir ./data_wsis_20x/Codipai_patch256_ostu/ \
--data_slide_dir ./data_wsis_20x/ \
--csv_path ./data_wsis_20x/Codipai_patch256_ostu/process_list_autogen.csv \
--feat_dir ./data_wsis_20x/Codipai_patch256_ostu/feats_ft_512 \
--batch_size 512 \
--slide_ext .tiff \
--ft_backbone_path './results/FT_codipai_res50_512_s1/sflod_0_checkpoint_backbone.pt' \
--target_patch_size 256