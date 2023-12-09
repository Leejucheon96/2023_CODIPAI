path_original_csv = '/data3/CMCS/raw_data/train_test_split_fold1.csv'
path_original_csv_label = '/data3/CMCS/raw_data/codipai_dataset_info.csv'
path_feats = '/data3/CMCS/raw_data_20x/Codipai_patch256_ostu/feats/h5_files/'
import os
import glob
ori_csv_contents = [item.split(',') for item in open(path_original_csv, 'r').read().split('\n')][1:-1]
ori_csv_contents_label = [item.split(',') for item in open(path_original_csv_label, 'r').read().split('\n')][1:-1]
current_feats = [os.path.basename(item).split('.')[0] for item in glob.glob(os.path.join(path_feats, '*'))]

train_test_dict = dict()
for item in ori_csv_contents:
    train_test_dict[item[0]] = item[1]
    # count_dict[item[1]] +=1

slide_id_label = dict()
for item in ori_csv_contents_label:
    slide_id_label[item[0]] = item[1]

count_dict = {
    "train": 0,
    "valid": 0
}
for item in current_feats:
    if item in current_feats:
        count_dict[train_test_dict[item]] += 1
count_label = {
    "train": {0:0, 1:0},
    "valid": {0:0, 1:0},
}

f_total = open('/data3/CMCS/raw_data_20x/codipai_dataset_info_partial_train_valid.csv', 'w')
f_total.write('slide_id,split\n')
f_info = open('/data3/CMCS/raw_data_20x/codipai_dataset_info_partial.csv', 'w')
f_info.write('slide_id,label\n')

f = open('/data3/CMCS/raw_data_20x/codipai_dataset_info_partial_train.csv', 'w')
f.write("slide_id,label\n")
for item in current_feats:
    if item in current_feats:
        if train_test_dict[item] == 'train':
            f.write(item + ',' + slide_id_label[item] + '\n')
            f_total.write(item + ',' + train_test_dict[item] + '\n')
            f_info.write(item + ',' + slide_id_label[item] + '\n')
            count_label[train_test_dict[item]][int(slide_id_label[item])] += 1
f.close()

f = open('/data3/CMCS/raw_data_20x/codipai_dataset_info_partial_valid.csv', 'w')
f.write("slide_id,label\n")
for item in current_feats:
    if item in current_feats:
        if train_test_dict[item] == 'valid':
            f.write(item + ',' + slide_id_label[item] + '\n')
            f_total.write(item + ',' + train_test_dict[item] + '\n')
            f_info.write(item + ',' + slide_id_label[item] + '\n')
            count_label[train_test_dict[item]][int(slide_id_label[item])] += 1

f.close()
f_total.close()
f_info.close()
print(count_dict)
print(count_label)