import tensorflow as tf
import cv2
import parameter as params

# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_Model(training):

    inputs = tf.keras.Input(shape=(params.img_w, params.img_h, 1), name='the_inputs')

    # Convolution layer (VGG)
    inner = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
        inputs)  # (None, 128, 64, 64)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)
    inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
        inner)  # (None, 64, 32, 128)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)
    inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
        inner)  # (None, 32, 16, 256)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)
    inner = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
        inner)  # (None, 32, 16, 256)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)
    inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
        inner)  # (None, 32, 8, 512)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)
    inner = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)
    inner = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), name='max4')(inner)  # (None, 32, 4, 512)

    inner = tf.keras.layers.Conv2D(256, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
        inner)  # (None, 32, 4, 512)
    inner = tf.keras.layers.BatchNormalization()(inner)
    inner = tf.keras.layers.Activation('relu')(inner)

    # CNN to RNN
    inner = tf.keras.layers.Reshape(target_shape=((32, 1024)), name='reshape')(inner)  # (None, 32, 2048)
    inner = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal', name='dense1')(
        inner)  # (None, 32, 64)

    # RNN layer
    lstm_1 = tf.keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
        inner)  # (None, 32, 512)
    lstm_1b = tf.keras.layers.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                                   name='lstm1_b')(inner)
    lstm1_merged = tf.keras.layers.add([lstm_1, lstm_1b])  # (None, 32, 512)
    lstm1_merged = tf.keras.layers.BatchNormalization()(lstm1_merged)
    lstm_2 = tf.keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(
        lstm1_merged)
    lstm_2b = tf.keras.layers.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                                   name='lstm2_b')(
        lstm1_merged)
    lstm2_merged = tf.keras.layers.concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)
    lstm_merged = tf.keras.layers.BatchNormalization()(lstm2_merged)

    # transforms RNN output to character activations:
    inner = tf.keras.layers.Dense(params.num_classes, kernel_initializer='he_normal', name='dense2')(
        lstm2_merged)  # (None, 32, 63)
    y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner)

    labels = tf.keras.layers.Input(name='the_labels', shape=[params.max_text_len], dtype='float32')  # (None ,8)
    input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
    label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    # loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)
    loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])  # (None, 1)
    # loss_out = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    if training:
        return tf.keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return tf.keras.Model(inputs=[inputs], outputs=y_pred)


if __name__=='__main__':

    crnn_model = get_Model(training=True)
    print(crnn_model.summary())

