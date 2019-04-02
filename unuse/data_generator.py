import cv2
import os, random
import numpy as np
from unuse import parameter as params
import json
from keras.applications.imagenet_utils import preprocess_input



class DataGenerator:
    def __init__(self,
                 img_dirpath,
                 json_path,
                 char_path,
                 img_w,
                 img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=params.max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)  # images list
        self.n = 0  # number of images
        self.indexes = []
        self.json_path = json_path
        self.char_path = char_path
        self.cur_index = 0
        self.imgs = []
        self.texts = []
        self.char_to_id = {}
        self.id_to_char = {}

    ## samples
    def build_data(self):
        print("DataGenerator, build data ...")
        # load image_label_dict, {image_name:label}
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.img_text_dict = json.load(f)
        self.img_fn_list = [i for i, j in self.img_text_dict.items()]
        self.n = len(self.img_fn_list)
        print("sample size of current generator: ", self.n)
        self.indexes = list(range(self.n))
        random.shuffle(self.indexes)
        # read char_dict
        f = open(self.char_path, 'r', encoding='utf-8')
        chars = f.read()
        self.char_to_id = {j: i for i, j in enumerate(chars)}
        self.id_to_char = {i: j for i, j in enumerate(chars)}
        #---------------------------------------------------------------------
        

    def next_sample(self):  ## index max -> 0
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        # load one image and its label
        img_idx = self.indexes[self.cur_index]
        img_fn = self.img_fn_list[img_idx]
        #print(os.path.join(self.img_dirpath, img_fn))
        img = cv2.imread(os.path.join(self.img_dirpath, img_fn), cv2.IMREAD_GRAYSCALE)
        #print(img.shape)

        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32)
        # img = np.array((img / 255.0) * 2.0 - 1.0)
        img = preprocess_input(img, mode='tf')
        text = self.img_text_dict[img_fn]
        return img, text


    def next_batch(self):  ## batch size
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])  # (bs, 128, 64, 1)
            # TODO  Y_data ---> 0
            Y_data = np.ones([self.batch_size, self.max_text_len])  # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))  # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                # convert text to num_arr label
                Y_data[i, :len(text)] = [self.char_to_id[i] for i in text]
                label_length[i] = len(text)

            # dict
            inputs = {
                'the_inputs': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) ->  value = 30
                'label_length': label_length  # (bs, 1) ->  value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}  # (bs, 1) -> 0
            yield (inputs, outputs)




if __name__ == "__main__":
    json_train_path = '/data/data/crnn_train_data/json/meta.json'
    json_val_path = '/data/data/crnn_train_data/json/val.json'
    save_path = '/data/data/crnn_train_data/cut_images'
    key_path = '/data/data/crnn_train_data/key/keys.txt'

    train_data = DataGenerator(img_dirpath=save_path,
                               json_path=json_val_path,
                               char_path=key_path,
                               img_w=params.img_w,
                               img_h=params.img_h,
                               batch_size=params.batch_size,
                               downsample_factor=params.downsample_factor,
                               max_text_len=params.max_text_len)
    train_data.build_data()

    inputs, outputs = train_data.next_batch().__next__()
    # # print(inputs)
    print(inputs)
