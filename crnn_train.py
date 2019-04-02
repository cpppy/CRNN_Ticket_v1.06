import os
import numpy as np
import tensorflow as tf
from crnn_model import crnn_model
from global_configuration import config


def crnn_net(is_training, feature, label, batch_size, l_size):
    seq_len = l_size
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=seq_len,
                                     num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    imgs = tf.image.resize_images(feature, (32, l_size * 4), method=0)
    input_imgs = tf.cast(x=imgs, dtype=tf.float32)

    with tf.variable_scope('shadow', reuse=False):
        net_out, tensor_dict = shadownet.build_shadownet(inputdata=input_imgs)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=label, inputs=net_out,
                                         sequence_length=seq_len * np.ones(batch_size)))

    # lstm l2
    lstm_tv = tf.trainable_variables(scope='LSTMLayers')
    r_lambda = 0.001
    regularization_cost = r_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in lstm_tv])
    cost = cost + regularization_cost

    model_params = tf.trainable_variables()
    tower_grad = tf.gradients(cost, model_params)

    return cost, zip(tower_grad, model_params), net_out, tensor_dict, seq_len


def get_shadownet_fn(num_gpus, variable_strategy, num_workers):
    def my_model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        loss, gradvars, preds, tensor_dict, seq_len = crnn_net(
            is_training, features, labels, params.batch_size, params.l_size)

        # if (mode == tf.estimator.ModeKeys.TRAIN or
        #         mode == tf.estimator.ModeKeys.EVAL):
        #     loss = ...
        # else:
        #     loss = None
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     train_op = ...
        # else:
        #     train_op = None
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     predictions = ...
        # else:
        #     predictions = None
        # return tf.estimator.EstimatorSpec(
        #     mode=mode,
        #     predictions=predictions,
        #     loss=loss,
        #     train_op=train_op)

        global_step = tf.train.get_global_step()
        starter_learning_rate = params.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   params.decay_steps,
                                                   params.decay_rate)
        # TODO: optimizer
        # decoded, log_prob = tf.nn.ctc_beam_search_decoder(preds,
        #                                                   seq_len * np.ones(params.batch_size),  # TODO
        #                                                   merge_repeated=False)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=optimizer)

    return my_model_fn


def my_model_fn(num_gpus, run_config):
    variable_strategy = 'CPU'
    model_fn = get_shadownet_fn(num_gpus,
                                variable_strategy,
                                run_config.num_worker_replicas or 1)

    return model_fn


def main():
    model_fn = my_model_fn(num_gpus=0, run_config=None)


if __name__ == '__main__':
    main()
