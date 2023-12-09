import pyvips
import os
import glob
import openslide
from PIL import Image
import tqdm
import warnings
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

path_raw_slides = './data_wsis/'
path_raw_slides_20x = './data_wsis_20x/'

if not os.path.exists(path_raw_slides_20x):
    os.mkdir(path_raw_slides_20x)

path_cancer = os.path.join(path_raw_slides, "Lymph_node_metasis_present")
path_non_cancer = os.path.join(path_raw_slides, "Lymph_node_metasis_absent")

list_cancer_slide_paths = glob.glob(os.path.join(path_cancer, '*'))
list_non_cancer_slide_paths = glob.glob(os.path.join(path_non_cancer, '*'))

for paths, cancer in zip([list_cancer_slide_paths, list_non_cancer_slide_paths], ["Lymph_node_metasis_present", "Lymph_node_metasis_absent"]):
    saved_dir = os.path.join(path_raw_slides_20x, cancer)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    for path in tqdm.tqdm(paths):
        try:
            img = pyvips.Image.tiffload(path)
            img.resize(0.5).tiffsave(saved_dir + '/' + os.path.basename(path))
        except:
            print("Err", path)
