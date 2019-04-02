"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
import numpy as np
import tensorflow as tf
import os
import tensorflow as tf
import re
from data_prepare import char_dict

def sparse_tensor_to_str(spares_tensor: tf.SparseTensor):
    """
    :param spares_tensor:
    :return: a str
    """
    indices = spares_tensor.indices
    values = spares_tensor.values
    # values = np.array([self.__ord_2_index_map[str(tmp)] for tmp in values])
    dense_shape = spares_tensor.dense_shape

    number_lists = np.zeros(dense_shape, dtype=values.dtype)
    str_lists = []
    res = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([char_dict.int_to_char[val] for val in number_list])
    for str_list in str_lists:
        res.append(''.join(c for c in str_list if c != '*'))
    return res



def read_features(tfrecords_dir, num_epochs, flag):
    """
    :param tfrecords_dir:
    :param num_epochs:
    :param flag: 'Train', 'Test', 'Validation'
    :return:
    """

    assert os.path.exists(tfrecords_dir)

    if not isinstance(flag, str):
        raise ValueError('flag should be a str in [\'Train\', \'Test\', \'Val\']')
    if flag.lower() not in ['train', 'test', 'val']:
        raise ValueError('flag should be a str in [\'Train\', \'Test\', \'Val\']')

    if flag.lower() == 'train':
        re_patten = r'^train_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
    elif flag.lower() == 'test':
        re_patten = r'^test_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
    else:
        re_patten = r'^val_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'

    tfrecords_list = [os.path.join(tfrecords_dir, tmp) for tmp in os.listdir(tfrecords_dir) if re.match(re_patten, tmp)]

    print('tfrecords_list: ', tfrecords_list)

    filename_queue = tf.train.string_input_producer(tfrecords_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature((), tf.string),
                                           'imagenames': tf.FixedLenFeature([1], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           # 'labels': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['images'], tf.uint8)
    images = tf.reshape(image, [256, 32])
    labels = features['labels']
    # labels = tf.one_hot(indices=labels, depth=10) #config.cfg.TRAIN.CLASSES_NUMS)
    labels = tf.cast(labels, tf.int32)
    # labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    imagenames = features['imagenames']
    return images, labels, imagenames


if __name__ == '__main__':

    imgs, labels, img_names = read_features('/data/data/crnn_tfrecords',
                                            num_epochs=None,
                                            flag='Train')

    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(tensors=[imgs, labels, img_names],
                                                                       batch_size=4,
                                                                       capacity=32 + 4 * 4,
                                                                       min_after_dequeue=32,
                                                                       num_threads=1)

    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    with sess.as_default():
        imgs_val, labels_val, img_names_val = sess.run([inputdata, input_labels, input_imagenames])
        print(type(labels_val))
        print(labels_val)
        print(img_names_val)
        scene_labels = sparse_tensor_to_str(labels_val)
        print('scene_labels_len: ', len(scene_labels))
        for index, scene_label in enumerate(scene_labels):
            img_name = img_names_val[index][0].decode('utf-8')
            print('{:s} --- {:s}'.format(img_name, scene_label))

        coord.request_stop()
        coord.join(threads=threads)
