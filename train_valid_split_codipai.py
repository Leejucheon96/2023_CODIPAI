import os
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

path_raw_slides = '/data3/CMCS/raw_data'
df_split_path = '/data3/CMCS/raw_data/train_test_split_fold1.csv'
df_info_path = '/data3/CMCS/raw_data/codipai_dataset_info.csv'

path_cancer = os.path.join(path_raw_slides, "Lymph_node_metasis_present")
path_non_cancer = os.path.join(path_raw_slides, "Lymph_node_metasis_absent")

list_cancer_slide_paths = glob.glob(os.path.join(path_cancer, '*'))
list_non_cancer_slide_paths = glob.glob(os.path.join(path_non_cancer, '*'))

train_cancer_paths, valid_cancer_paths = train_test_split(list_cancer_slide_paths, test_size=0.2, random_state=42)
train_non_cancer_paths, valid_non_cancer_paths = train_test_split(list_non_cancer_slide_paths, test_size=0.2, random_state=42)

### Creating csv file for train-valid split
f = open(df_split_path, 'w')
f.write("file_name,split\n")
for split, paths in zip(['train', 'valid'], [[train_cancer_paths, train_non_cancer_paths], [valid_cancer_paths, valid_non_cancer_paths]]):
    print("Processing", split)
    for cancer_type, cancer_paths in zip(["cancer", "non_cancer"], paths):
        print("Cancer type", cancer_type)
        for path in tqdm(cancer_paths):
            f.write(os.path.basename(path).split('.')[0] + ',' + split + '\n')
f.close()

### Creating csv file for filename-label
df_ori_info_path = '/data3/CMCS/2023_dataton_age_sex_본선.csv'
contents = [item.split(',') for item in open(df_ori_info_path, 'r').read().split('\n')][1:-1]
f = open(df_info_path, 'w')
f.write("slide_id,label\n")

dict_map_label = {
    '비전이': 0,
    '전이': 1
}

for item in contents:
    f.write(item[0] + ',' + str(dict_map_label[item[1]]) + '\n')
f.close()