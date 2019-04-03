import tensorflow as tf
import tqdm
import os
import json
import cv2
import math
import numpy as np
from data_prepare import char_dict

# # load char_dict
# key_path = '/data/CRNN_Ticket_v1.06/key/keys.txt'
# char_to_int = {}
# int_to_char = {}
# with open(key_path, 'r', encoding='utf-8') as key_f:
#     chars = key_f.read()
#     for idx, char in enumerate(chars):
#         char_to_int[char] = idx
#         int_to_char[idx] = char
#     num_classes = len(chars) + 1


def encode_labels(labels):
    """
        encode the labels for ctc loss
    :param labels:
    :return:
    """
    encoded_labeles = []
    lengths = []
    for label in labels:
        encode_label = [char_dict.char_to_int[char] for char in label]
        encoded_labeles.append(encode_label)
        lengths.append(len(label))
    return encoded_labeles, lengths


def int64_feature(value):
    """
        Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """
        Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_features(tfrecords_path,
                   labels,
                   images,
                   imagenames):
    """

    :param tfrecords_path:
    :param labels:
    :param images:
    :param imagenames:
    :return:
    """
    assert len(labels) == len(images) == len(imagenames)

    # print(labels[0])
    # print(len(labels[0]))
    labels, length = encode_labels(labels)

    if not os.path.exists(os.path.split(tfrecords_path)[0]):
        os.makedirs(os.path.split(tfrecords_path)[0])

    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for index, image in enumerate(images):
            features = tf.train.Features(feature={
                # 'labels': int64_feature(labels[index]),
                'labels': int64_feature(labels[index]),
                'images': bytes_feature(image),
                'imagenames': bytes_feature(imagenames[index])
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            # sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index+1, len(images), imagenames[index]))
            # sys.stdout.flush()
        # sys.stdout.write('\n')
        # sys.stdout.flush()
    return


def gen_tfrecords(img_dir, json_fpath, batch_size, save_dir, data_type):
    # write train tfrecords
    print('Start generating tf records for %s' % data_type)

    # load json
    with open(json_fpath, 'r', encoding='utf-8') as train_f:
        img_text_dict = json.load(train_f)
        # print(img_text_dict)
        img_fn_list = [img_fn for img_fn, text in img_text_dict.items()]
        img_path_list = [os.path.join(img_dir, img_fn) for img_fn in img_fn_list]
        text_list = [text for img_fn, text in img_text_dict.items()]

    train_images_nums = len(img_fn_list)
    # epoch_nums = int(math.ceil(train_images_nums / batch_size))
    epoch_nums = int(train_images_nums / batch_size)
    for loop in tqdm.tqdm(range(epoch_nums)):
        # TODO: normalization
        loop_fn_list = img_fn_list[loop * batch_size: (loop + 1) * batch_size]
        loop_path_list = img_path_list[loop * batch_size: (loop + 1) * batch_size]
        loop_text_list = text_list[loop * batch_size: (loop + 1) * batch_size]

        train_images = [cv2.resize(cv2.imread(img_path, 0), (256, 32)) for img_path in loop_path_list]
        train_images = np.asarray(train_images)
        train_labels = np.asarray(loop_text_list)
        train_imagenames = np.asarray(loop_fn_list)
        print('image_shape: ', train_images[0].shape)
        print("labels_0: ", train_labels[0])
        print("image_name_0: ", train_imagenames[0])
        # TODO: ch1 --> ch3
        train_images = [bytes(list(np.reshape(tmp, [256 * 32]))) for tmp in train_images]

        if loop * batch_size + batch_size > train_images_nums:
            save_tfrecord_path = os.path.join(save_dir, '{:s}_feature_{:d}_{:d}.tfrecords'.format(
                data_type, loop * batch_size, train_images_nums))
        else:
            save_tfrecord_path = os.path.join(save_dir, '{:s}_feature_{:d}_{:d}.tfrecords'.format(
                data_type, loop * batch_size, loop * batch_size + batch_size))
        print('%s_tfrecord_path: ' % data_type, save_tfrecord_path)

        write_features(tfrecords_path=save_tfrecord_path,
                       labels=train_labels,
                       images=train_images,
                       imagenames=train_imagenames)


def main():
    img_dir = '/data/data/crnn_train_data/images'
    train_json_path = '/data/data/crnn_train_data/labels/train.json'
    val_json_path = '/data/data/crnn_train_data/labels/val.json'
    batch_size = 8
    save_dir = '/data/data/crnn_tfrecords'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # train_data
    gen_tfrecords(img_dir=img_dir,
                  json_fpath=train_json_path,
                  batch_size=batch_size,
                  save_dir=save_dir,
                  data_type='train')
    # val_data
    gen_tfrecords(img_dir=img_dir,
                  json_fpath=val_json_path,
                  batch_size=batch_size,
                  save_dir=save_dir,
                  data_type='val')
    # test_data
    gen_tfrecords(img_dir=img_dir,
                  json_fpath=val_json_path,
                  batch_size=batch_size,
                  save_dir=save_dir,
                  data_type='test')


if __name__ == '__main__':
    main()

    # -----------------------------------------------------------------------------------
    '''
    # write train tfrecords
    print('Start writing validation tf records')

    # load json
    with open(val_json_path, 'r', encoding='utf-8') as train_f:
        img_text_dict = json.load(train_f)
        # print(img_text_dict)
        img_fn_list = [img_fn for img_fn, text in img_text_dict.items()]
        img_path_list = [os.path.join(img_dir, img_fn) for img_fn in img_fn_list]
        text_list = [text for img_fn, text in img_text_dict.items()]

    val_images_nums = len(img_fn_list)
    epoch_nums = int(math.ceil(val_images_nums / batch_size))
    # epoch_nums = int(train_images_nums / batch_size)
    for loop in tqdm.tqdm(range(epoch_nums)):
        # TODO: normalization
        loop_fn_list = img_fn_list[loop * batch_size: (loop + 1) * batch_size]
        loop_path_list = img_path_list[loop * batch_size: (loop + 1) * batch_size]
        loop_text_list = text_list[loop * batch_size: (loop + 1) * batch_size]

        val_images = [cv2.resize(cv2.imread(img_path, 0), (32, 32)) for img_path in loop_path_list]
        val_images = np.asarray(val_images)
        val_labels = np.asarray(loop_text_list)
        val_imagenames = np.asarray(loop_fn_list)
        # print('train_image_shape: ', train_images[0].shape)
        # print("train_labels_0: ", train_labels[0])
        # print("train_imagenames: ", train_imagenames[0])
        # TODO: ch1 --> ch3
        val_images = [bytes(list(np.reshape(tmp, [32 * 32]))) for tmp in val_images]

        if loop * batch_size + batch_size > val_images_nums:
            val_tfrecord_path = os.path.join(save_dir, 'val_feature_{:d}_{:d}.tfrecords'.format(
                loop * batch_size, val_images_nums))
        else:
            val_tfrecord_path = os.path.join(save_dir, 'val_feature_{:d}_{:d}.tfrecords'.format(
                loop * batch_size, loop * batch_size + batch_size))
        print('val_tfrecord_path: ', val_tfrecord_path)

        write_features(tfrecords_path=val_tfrecord_path,
                       labels=val_labels,
                       images=val_images,
                       imagenames=val_imagenames)
    '''
