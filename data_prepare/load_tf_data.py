"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
import numpy as np
import tensorflow as tf
import os
import os.path as ops
import re



def read_features(tfrecords_dir, num_epochs, flag):
    """

    :param tfrecords_dir:
    :param num_epochs:
    :param flag: 'Train', 'Test', 'Validation'
    :return:
    """

    assert ops.exists(tfrecords_dir)

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

    tfrecords_list = [ops.join(tfrecords_dir, tmp) for tmp in os.listdir(tfrecords_dir) if re.match(re_patten, tmp)]

    print('tfrecords_list: ', tfrecords_list)

    filename_queue = tf.train.string_input_producer(tfrecords_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature((), tf.string),
                                           'imagenames': tf.FixedLenFeature([1], tf.string),
                                           # 'labels': tf.VarLenFeature(tf.int64),
                                           'labels': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['images'], tf.uint8)
    images = tf.reshape(image, [32, 32])
    labels = features['labels']
    labels = tf.one_hot(indices=labels, depth=10) #config.cfg.TRAIN.CLASSES_NUMS)
    labels = tf.cast(labels, tf.int32)
    # labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    imagenames = features['imagenames']
    return images, labels, imagenames


if __name__ == '__main__':

    imgs, labels, img_names = read_features('/data/data/mnist_tfrecords',
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
        # print('gt_labels_len: ', len(gt_labels))
        # for index, gt_label in enumerate(gt_labels):
        #     name = img_names_val[index][0].decode('utf-8')
        #     name = ops.split(name)[1]
        #     name = name.split('_')[-2]
        #     print('{:s} --- {:s}'.format(gt_label, name))

        coord.request_stop()
        coord.join(threads=threads)
