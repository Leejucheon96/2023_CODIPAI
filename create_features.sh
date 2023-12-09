CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
--data_h5_dir /data3/CMCS/raw_data/Codipai_patch256_ostu/ \
--data_slide_dir /data3/CMCS/raw_data/ \
--csv_path /data3/CMCS/raw_data/Codipai_patch256_ostu/process_list_autogen.csv \
--feat_dir /data3/CMCS/raw_data/Codipai_patch256_ostu/feats \
--batch_size 512 \
--slide_ext .tiff