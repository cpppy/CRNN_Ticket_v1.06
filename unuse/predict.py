import itertools

import cv2
import numpy as np
from keras import backend as K

from unuse import Model as crnn_model, parameter as params

K.set_learning_phase(0)


def get_char_id_dict(char_path):
    f = open(char_path, 'r', encoding='utf-8')
    chars = f.read()
    char_to_id = {j: i for i, j in enumerate(chars)}
    id_to_char = {i: j for i, j in enumerate(chars)}
    return char_to_id, id_to_char

def decode_label(out):
    # out : (1, 32, 42)
    # out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = list(np.argmax(out[0, :], axis=1))
    # print("out_best: ", out_best)
    # gp_res = itertools.groupby(out_best)

    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

    outstr = ''
    char_to_id, id_to_char = get_char_id_dict(params.char_path)
    print(id_to_char)
    for i in out_best:
        if i < len(char_to_id):
            outstr += id_to_char[i]
    return outstr



if __name__ == "__main__":

    # Get CRNN model
    crnn_model = crnn_model.get_Model(training=False)
    # Load weights
    latest_weights = '/data/CRNN_draft/save_model/output_multi_CRNN_2nd_at_epoch_3.h5'
    print("latest_weights: ", latest_weights)
    try:
        if latest_weights != None:
            # model.load_weights(os.path.join(weights_dir, latest_weights))
            crnn_model.load_weights(latest_weights)
            print("...load exist model or weights: ", latest_weights)
        else:
            print("history weights file not exist, stop predicting.")
            exit(1)
    except:
        print('fail to load weights file, stop predicting.')
        exit(1)

    test_img = 'train_data/cut_images/196.jpg'
    print('test_img: ', test_img)
    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)


    img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)

    net_out_value = crnn_model.predict(img_pred)
    print("net_output: ", net_out_value.shape)

    pred_texts = decode_label(net_out_value)
    print("pred_texts: ", pred_texts)

