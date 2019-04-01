import json
import math
import os
import random
import shutil
import tarfile

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.LOAD_TRUNCATED_IMAGES = True

origin_data_dir = '/data/data/train_500'


class DataProcess(object):

    def __init__(self, thread_id, head_index, tail_index):

        self.thread_id = thread_id
        self.begin_index = head_index
        self.end_index = tail_index

        self.image_path = origin_data_dir + '/images'
        self.text_path = origin_data_dir + '/labels'
        self.tmp_file_label_dict = {}
        self.json_path = '/data/data/crnn_train_data/json/meta_%d.json' % self.thread_id
        self.json_val_path = '/data/data/crnn_train_data/json/val_%d.json' % self.thread_id
        self.save_path = '/data/data/crnn_train_data/cut_images'
        self.key_path = '/data/data/crnn_train_data/key/keys.txt'
        self.output_key_path = '/data/output/CRNN_draft/keys.txt'

        self.high_frequency_labels = ['姓名',
                                      '性别',
                                      '男',
                                      '女',
                                      '民族汉',
                                      '出生',
                                      '住址',
                                      '公民身份证号码',
                                      '签发机关',
                                      '有效期限']
        self.un_sampling_labels = ['中华人民共和国', '居民身份证', '###', ' ', '']

        self.high_freq_sampling_count = {label: 0 for label in self.high_frequency_labels}

        self.high_freq_limit = int((tail_index - head_index) * 0.1)

        self.validation_split_ratio = 0.1

        self.img_height = 64
        self.img_width = 128

        # init directory of train_data
        if not os.path.exists('/data/data/crnn_train_data'):
            os.mkdir('/data/data/crnn_train_data')
        if not os.path.exists('/data/data/crnn_train_data/json'):
            os.mkdir('/data/data/crnn_train_data/json')
        if not os.path.exists('/data/data/crnn_train_data/key'):
            os.mkdir('/data/data/crnn_train_data/key')
        if not os.path.exists('/data/data/crnn_train_data/cut_images'):
            os.mkdir('/data/data/crnn_train_data/cut_images')
        if not os.path.exists('/data/output'):
            os.mkdir('/data/output')
        if not os.path.exists('/data/output/CRNN_draft'):
            os.mkdir('/data/output/CRNN_draft')

    def data_preprocess(self):
        # load data from tarfile
        if not os.path.exists(self.image_path):
            print("thread_%d" % self.thread_id, "train_data not exist, task over")
            return
        else:
            print("thread_%d" % self.thread_id, "data(images&labels) already exists.")

        # do preprocess
        valid_img_count = 0
        dic = {}
        # max_height, min_height, max_label_length = -1, -1, -1
        # max_label_length_file = ""
        # max_height_file = ""
        print("thread_%d" % self.thread_id, 'segmentation task, start ... ')
        for root, dirs, files in os.walk(self.image_path):
            files_for_current_thread = files[self.begin_index: self.end_index]
            pbar = tqdm(total=len(files_for_current_thread), desc="thread_%d" % self.thread_id, )
            for file in files_for_current_thread:
                pbar.update(1)
                # file_name_list.append(os.path.splitext(file)[0])
                file_name_head = os.path.splitext(file)[0]
                # print(file_name_head)
                text_file = os.path.join(self.text_path, file_name_head + ".txt")
                # print(text_file)
                try:
                    raw_img = Image.open(os.path.join(self.image_path, file))
                except Exception as e:
                    print("thread_%d" % self.thread_id, "warn: fail to read image: ",
                          os.path.join(self.image_path, file))
                    continue
                with open(text_file, encoding='utf-8') as f:
                    for labels in f.readlines():
                        labels = labels.strip('\r\n')
                        arr = labels.split(',')

                        x1 = float(arr[0])  # 左上角
                        y1 = float(arr[1])
                        x2 = float(arr[2])  # 左下角
                        y2 = float(arr[3])
                        x3 = float(arr[4])  # 右下角
                        y3 = float(arr[5])
                        x4 = float(arr[6])  # 右上角
                        y4 = float(arr[7])

                        # do_extend
                        enlarge_ratio = round(40*np.abs(x4-x1)/850.0)
                        x1 -= random.randint(0,enlarge_ratio)
                        x2 -= random.randint(0,enlarge_ratio)
                        x3 += random.randint(0,enlarge_ratio)
                        x4 += random.randint(0,enlarge_ratio)

                        y1 -= random.randint(0,enlarge_ratio)
                        y2 += random.randint(0,enlarge_ratio)
                        y3 += random.randint(0,enlarge_ratio)
                        y4 -= random.randint(0,enlarge_ratio)

                        label = arr[8]
                        label_length = len(label)

                        if label in self.un_sampling_labels:
                            continue
                        if label in self.high_frequency_labels:
                            if self.high_freq_sampling_count[label] > self.high_freq_limit:
                                continue
                            self.high_freq_sampling_count[label] += 1

                        left = int(x1 if x1 < x2 else x2)
                        top = int(y1 if y1 < y4 else y4)
                        right = math.ceil(x3 if x3 > x4 else x4)
                        bottom = math.ceil(y2 if y2 > y3 else y3)

                        if (left == right or top == bottom):
                            continue
                        temp = left
                        if (left > right):
                            left = right
                            right = temp
                        temp = top
                        if (top > bottom):
                            top = bottom
                            bottom = temp
                        box = (left, top, right, bottom)

                        # print(box)
                        # print(label_length)
                        height = bottom - top
                        width = right - left
                        aspect_ratio = height / width
                        # 剪裁图片
                        if (width > height or aspect_ratio <= 1.5):
                            img = self.horizontal_image_method(raw_img, box)
                        else:
                            img = self.vertical_image_method(raw_img, box, label_length)

                        # # 获取最大和最小height
                        # width, height = img.size
                        # if (i == 1):
                        #     min_height = height
                        # if (max_height < height):
                        #     max_height = height
                        #     max_height_file = text_file
                        # if (min_height > height):
                        #     min_height = height
                        # if (max_label_length < label_length):
                        #     max_label_length = label_length
                        #     max_label_length_file = text_file

                        # save image
                        valid_img_count += 1
                        image_name = '%d_' % self.thread_id + str(valid_img_count) + ".jpg"
                        img.save(os.path.join(self.save_path, image_name), quality=100)

                        dic[image_name] = str(label)
            pbar.close()
        json_file = open(self.json_path, 'w', encoding='utf-8')
        json.dump(dic, json_file, ensure_ascii=False)
        json_file.close()

    # uncompress data from tarfile
    def uncompress_tarfile(self, tarfile_path, save_path):
        tar = tarfile.open(tarfile_path)
        tar.extractall(path=save_path)
        tar.close()

    # 处理横的训练集
    def horizontal_image_method(self, img, box):
        region = img.crop(box)
        region = region.convert('L')
        return region

    # 处理竖的训练集
    def vertical_image_method(self, img, box, label_length):
        region = img.crop(box)
        region = region.convert('L')

        # 将竖的图片转成横的图片
        region_width, region_height = region.size
        one_word_height = int(region_height / label_length)
        temp_region_left = 0
        if (one_word_height == 0):
            return img

        new_img = Image.new('L', (region_width * label_length, one_word_height))

        for index in range(label_length):
            temp_region_height = one_word_height * index
            temp_region = region.crop((0, temp_region_height, region_width, temp_region_height + one_word_height))
            new_img.paste(temp_region, (temp_region_left, 0, temp_region_left + region_width, one_word_height))
            temp_region_left += region_width
        return new_img

    # 缩放图片
    def rescale(self):
        for root, dirs, files in os.walk(self.image_path):
            for file in files:
                img = Image.open(os.path.join(self.image_path, file))
                img = img.resize((self.img_width, self.img_height), Image.ANTIALIAS)
                save_path = os.path.join(self.save_path, file)
                print(save_path)
                img = img.convert("RGB")
                img.save(save_path, quality=100)

    # 获取所有字符集
    def generate_key(self):
        # check keys.txt
        if os.path.exists(self.output_key_path):
            print("keys.txt have been build, load file: ", self.output_key_path)
            shutil.copyfile(self.output_key_path, self.key_path)
        else:
            print("old key_file not exist, generating a new one ...")
            char = ""
            with open(self.json_path, 'r', encoding='utf-8') as f:
                image_label = json.load(f)
                labels = [j for i, j in image_label.items()]
                for label in labels:
                    label = str(label)
                    for i in label:
                        if (i in char):
                            continue
                        else:
                            char += i

                key_file = open(self.key_path, 'w', encoding='utf-8')
                key_file.write(char)
                key_file.close()

                # output keys.txt to /data/output/CRNN/
                key_file = open(self.output_key_path, 'w', encoding='utf-8')
                key_file.write(char)
                key_file.close()

    # 随机取一部分数据为测试集
    def random_get_val(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            image_label = json.load(f)
            image_file = [i for i, j in image_label.items()]
            # 所有训练集的长度
            nums = len(image_file)
            # print("image_file_length: ", len(image_file))
            # resultList = random.sample(range(1, nums + 1), 3000)
            # select some sample as validation data
            val_size = int(nums * self.validation_split_ratio)
            resultList = random.sample(range(1, nums + 1), val_size)
            dic = {}
            for num in resultList:
                image_name = str(num) + '.jpg'
                dic[image_name] = image_label[image_name]
                image_label.pop(image_name)
            self.train_length = len(image_label)  # remain samples
            self.val_length = len(dic)  # selected samples
            json_val_file = open(self.json_val_path, 'w', encoding='utf-8')
            json.dump(dic, json_val_file, ensure_ascii=False)
            json_val_file.close()
            json_meta_file = open(self.json_path, 'w', encoding='utf-8')
            json.dump(image_label, json_meta_file, ensure_ascii=False)
            json_meta_file.close()

    # generate a batch size of data
    def generate(self,
                 json_path,
                 image_path,
                 char_path,
                 batch_size,
                 max_label_length,
                 image_size):
        with open(json_path, 'r', encoding='utf-8') as f:
            image_label = json.load(f)
        f = open(char_path, 'r', encoding='utf-8')
        char = f.read()
        char_to_id = {j: i for i, j in enumerate(char)}
        id_to_char = {i: j for i, j in enumerate(char)}
        image_file = [i for i, j in image_label.items()]

        x = np.zeros((batch_size, image_size[0], image_size[1], 1), dtype=np.float)
        labels = np.ones([batch_size, max_label_length]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])

        r_n = random_uniform_num(len(image_file))

        image_file = np.array(image_file)
        while 1:
            shuffle_image = image_file[r_n.get(batch_size)]
            for i, j in enumerate(shuffle_image):
                img = Image.open(image_path + j)
                img_arr = np.array(img, 'f') / 255.0 - 0.5
                # resize, height-->32
                transform_ratio = float(img.width) / float(image_size[0])
                # new_width = int(img.height // transform_ratio)
                new_width = image_size[1]
                img_arr = cv2.resize(img_arr, (new_width, image_size[0]))

                x[i, :, :new_width] = np.expand_dims(img_arr, axis=2)
                label = image_label[j]

                label_length[i] = len(label)
                input_length[i] = image_size[1] // 4 + 1
                labels[i, :len(label)] = [char_to_id[i] for i in label]
            inputs = {'the_input': x,
                      'the_labels': labels,
                      'input_length': input_length,
                      'label_length': label_length
                      }
            outputs = {'ctc': np.zeros([batch_size])}
            yield (inputs, outputs)


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batch_size):
        r_n = []
        if (self.index + batch_size > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batch_size) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)

        else:
            r_n = self.range[self.index:self.index + batch_size]
            self.index = self.index + batch_size
        return r_n


if __name__ == "__main__":
    data_proc = DataProcess()
    data_proc.data_preprocess()
    data_proc.generate_key()
    data_proc.random_get_val()
    print("segmentation task, over !")
