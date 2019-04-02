import os
# -------------- project info -----------------------
proj_root_path = '/data/CRNN_Ticket_v1.06'
#load_weights_path = os.path.join(proj_root_path, 'save_model/output_multi_CRNN_5th_plus_epoch_18.h5'
load_weights_path = None
# load_weights_path = '/data/output/crnn_ticket_20190327/crnn_weights_d10w_ticket_id_date_20190327_ep_1.h5'
test_weights_path = load_weights_path

weights_save_path = '/data/output/crnn_ticket_20190327'
if not os.path.exists(weights_save_path):
    os.mkdir(weights_save_path)


# --------------- load origin data -------------------
origin_data_dir = '/data/data/'
tarfile_path = '/data/data/ticket_train_data_10w_ticket_id_date_20190327_init.tar'
uncompress_dir_path = '/data/data/ticket_train_data'

origin_images_path = uncompress_dir_path + '/images'
origin_labels_path = uncompress_dir_path + '/labels'


# --------------- train data preprocess -----------------

# num_classes = len(letters) + 1
num_classes = 17   # default, the real value based on the string lenght in keys.txt

validation_split_ratio = 0.1

# chinese & alphabet & number
json_train_path = '/data/data/ticket_train_data/labels/train.json'
json_val_path = '/data/data/ticket_train_data/labels/val.json'
cut_img_save_path = '/data/data/ticket_train_data/images'
char_path = proj_root_path + '/key/keys.txt'

generate_key_path = char_path
output_key_path = char_path

gpu_nums_in_multi_model = 4

# ---------------model params ---------------------------

img_w, img_h = 256, 32

# Network parameters
batch_size = 16
val_batch_size = 16
epoch_num = 20

downsample_factor = 8
max_text_len = 12
