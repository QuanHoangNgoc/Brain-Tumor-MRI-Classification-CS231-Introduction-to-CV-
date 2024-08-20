
import os
STORE_ALL_DATA_PUT_ALL = False
CNT_EXTRACT = -1
NAME = ['complex_fd', 'org_image', 'enh_image', 'gabor', 'glcm', 'hog_fd',
        'hog_image', 'lbp_fd', 'lbp_image', 'emw_fd', 'e_image', 'w_image']


CLEAR_LOG = False  # clear file log(if need)

LABEL = os.listdir('D:\cd_data_C\Desktop\git_upload\MRI\Dataset2\Training')
FOLDER_DATASET = 'Dataset2'


FEATURE_PIPELINE = ([(0, 0)], [(3, 0)])  # (feature_id, feature_type)
