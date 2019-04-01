import tensorflow as tf
import tf_keras_Model as crnn_model






if __name__=='__main__':


    model_fn = crnn_model.get_Model(training=True)

    estimator = tf.keras.estimator.model_to_estimator(model_fn, model_dir='./save_model')























