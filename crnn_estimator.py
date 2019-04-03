import tensorflow as tf
from unuse import tf_keras_Model as crnn_model

from data_prepare.crnn_dataset import CrnnDataSet
from data_prepare import load_tf_data
import crnn_train

def my_input_fn(data_dir='/data/data/crnn_tfrecords',
                subset='Train',
                num_shards=0,
                batch_size=4,
                use_distortion_for_training=False):
    with tf.device('/cpu:0'):
        # use_distortion = subset == 'train' and use_distortion_for_training
        dataset = CrnnDataSet(data_dir, subset)
        input_data, input_labels = dataset.make_batch(batch_size)
        # labels = tf.one_hot(indices=input_labels, depth=10)  # config.cfg.TRAIN.CLASSES_NUMS)
        # one_hot_labels = tf.cast(labels, tf.int32)

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            num_shards = 1

        feature_shards = tf.split(input_data, num_shards)
        label_shards = tf.sparse_split(sp_input=input_labels, num_split=num_shards, axis=0)
        # label_shards = tf.split(input_labels, num_shards)
        return feature_shards[0], label_shards[0]


def check_my_input_fn():
    # crnn_dataset = CrnnDataSet(data_dir='/data/data/crnn_tfrecords',
    #                            subset='train')
    #
    # imgs, labels = crnn_dataset.make_batch(batch_size=32)
    feature_shards, label_shards = my_input_fn()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        _feature_shards, _label_shards = sess.run([feature_shards, label_shards])
        _imgs, _labels = _feature_shards[0], _label_shards[0]
        scene_labels = load_tf_data.sparse_tensor_to_str(_labels)
        print('scene_labels_len: ', len(scene_labels))
        for index, scene_label in enumerate(scene_labels):
            # img_name = img_names_val[index][0].decode('utf-8')
            # print('{:s} --- {:s}'.format(img_name, scene_label))
            print('scene_label: ', scene_label)

        coord.request_stop()
        coord.join(threads=threads)


def main():
    # train_model = crnn_model.get_Model(training=True)
    # train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
    #                     optimizer=tf.keras.optimizers.Adadelta()
    #                     )
    #
    # train_estimator = tf.keras.estimator.model_to_estimator(train_model, model_dir='/data/output/crnn_estimator_ckp')

    estimator = None

    BATCH_SIZE = 16
    EPOCHS = 5
    STEPS = 2000
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: my_input_fn(batch_size=BATCH_SIZE),
                                        max_steps=STEPS)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: my_input_fn(batch_size=BATCH_SIZE),
                                      steps=1,
                                      start_delay_secs=3
                                      )

    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec
                                    )


if __name__ == '__main__':
    check_my_input_fn()

    main()
