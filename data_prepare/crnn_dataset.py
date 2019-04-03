import os
import tensorflow as tf
import re
from data_prepare import load_tf_data


class CrnnDataSet(object):

    def __init__(self, data_dir, subset='train', use_distortion=False):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_tfrecord_fpath_list(self):

        if self.subset.lower() == 'train':
            re_patten = r'^train_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        elif self.subset.lower() == 'val':
            re_patten = r'^val_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        elif self.subset.lower() == 'test':
            re_patten = r'^test_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        else:
           raise ValueError('Invalid data subset "%s"' % self.subset)

        tfrecord_fpath_list = [os.path.join(self.data_dir, fn) for fn in os.listdir(self.data_dir) if re.match(re_patten, fn)]
        print('%s_tfrecord_list: '%self.subset.lower(), tfrecord_fpath_list)
        return tfrecord_fpath_list

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'images': tf.FixedLenFeature((), tf.string),
                'imagenames': tf.FixedLenFeature([1], tf.string),
                # 'labels': tf.FixedLenFeature((), tf.int64),
                'labels': tf.VarLenFeature(tf.int64),
            })
        images = tf.decode_raw(features['images'], tf.uint8)
        images = tf.reshape(images, [32, 100, 3])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        return images, labels


    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_tfrecord_fpath_list()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat()

        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=2)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = 32 #int(MnistDataSet.num_examples_per_epoch(self.subset) * 0.001)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 2 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 32
        elif subset == 'val':
            return 16
        elif subset == 'test':
            return 16
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


def main():

    crnn_dataset = CrnnDataSet(data_dir='/data/data/crnn_tfrecords',
                               subset='train')

    imgs, labels = crnn_dataset.make_batch(batch_size=32)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        _imgs, _labels = sess.run([imgs, labels])
        scene_labels = load_tf_data.sparse_tensor_to_str(_labels)
        print('scene_labels_len: ', len(scene_labels))
        for index, scene_label in enumerate(scene_labels):
            # img_name = img_names_val[index][0].decode('utf-8')
            # print('{:s} --- {:s}'.format(img_name, scene_label))
            print('scene_label: ', scene_label)

        coord.request_stop()
        coord.join(threads=threads)


if __name__=='__main__':
    main()