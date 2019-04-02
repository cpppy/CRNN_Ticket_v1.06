from unuse import segmentation_task
import os
import multiprocessing
import json
import shutil
import random


sample_num = 1000

pid_num = 4

origin_data_dir = '/data/data/train_500'
meta_json_dir='/data/data/crnn_train_data/json'
total_json_path = '/data/data/crnn_train_data/json/meta.json'
json_val_path = '/data/data/crnn_train_data/json/val.json'

curr_key_path = '/data/data/crnn_train_data/key/keys.txt'
output_key_path = '/data/output/CRNN_draft/keys.txt'

validation_split_ratio = 0.1


def worker(sign, thread_id, head_index, tail_index):
    print("thread_%d"%thread_id, 'process quantity: ', tail_index - head_index)

    data_proc = segmentation_task.DataProcess(thread_id=thread_id,
                                              head_index=head_index,
                                              tail_index=tail_index)

    data_proc.data_preprocess()
    print("thread_%d" % thread_id, "finish segmentation task !")





def do_multi_segmentation_task():

    images_path = os.path.join(origin_data_dir, 'images')

    image_list = os.listdir(images_path)
    # total_num = len(image_list)

    num_for_each = int(len(image_list)/pid_num)
    print("num for each pid: ", num_for_each)

    pid_list = []

    for i in range(pid_num):
        head_index = num_for_each * i
        tail_index = min(num_for_each * (i + 1), len(image_list))
        curr_pid = multiprocessing.Process(target=worker,
                                           args=('process',
                                                 i + 1,
                                                 head_index,
                                                 tail_index))
        pid_list.append(curr_pid)
    for pid in pid_list:
        pid.start()
        # pid.join()
    for pid in pid_list:
        pid.join()

        # wait all task over
    print('multi_pid task over')

    print('process meta_data...')
    combine_label_dict(meta_json_dir)
    generate_key()
    random_get_val()
    print("segmentation task, over !")

def combine_label_dict(meta_json_dir='/data/data/crnn_train_data/json'):
    meta_json_files = os.listdir(meta_json_dir)
    total_labels = {}
    for meta_json_file in meta_json_files:
        if 'meta_' in meta_json_file:
            with open(os.path.join(meta_json_dir, meta_json_file), 'r', encoding='utf-8') as meta_f:
                image_label = json.load(meta_f)
                total_labels = dict(total_labels, **image_label)
    json_file = open(total_json_path, 'w', encoding='utf-8')
    json.dump(total_labels, json_file, ensure_ascii=False)
    json_file.close()



def generate_key():
    # check keys.txt
    if os.path.exists(output_key_path):
        print("keys.txt have been build, load file: ", output_key_path)
        shutil.copyfile(output_key_path, curr_key_path)
    else:
        print("old key_file not exist, generating a new one ...")
        char = ""
        with open(total_json_path, 'r', encoding='utf-8') as f:
            image_label = json.load(f)
            labels = [j for i, j in image_label.items()]
            for label in labels:
                label = str(label)
                for i in label:
                    if (i in char):
                        continue
                    else:
                        char += i

            key_file = open(curr_key_path, 'w', encoding='utf-8')
            key_file.write(char)
            key_file.close()

            # output keys.txt to /data/output/CRNN/
            key_file = open(output_key_path, 'w', encoding='utf-8')
            key_file.write(char)
            key_file.close()


def random_get_val():
    with open(total_json_path, 'r', encoding='utf-8') as f:
        total_labels = json.load(f)
        image_files = [i for i, j in total_labels.items()]
        nums = len(image_files)
        val_size = int(nums * validation_split_ratio)
        val_choosed_indexes = random.sample(range(nums), val_size)
        val_labels = {}
        for val_index in val_choosed_indexes:
            val_labels[image_files[val_index]] = total_labels[image_files[val_index]]
            total_labels.pop(image_files[val_index])
        print("train_samples: ", len(total_labels))
        print("validation_samples: ", len(val_labels))
        json_val_file = open(json_val_path, 'w', encoding='utf-8')
        json.dump(val_labels, json_val_file, ensure_ascii=False)
        json_val_file.close()
        json_meta_file = open(total_json_path, 'w', encoding='utf-8')
        json.dump(total_labels, json_meta_file, ensure_ascii=False)
        json_meta_file.close()



if __name__ == "__main__":

    do_multi_segmentation_task()
