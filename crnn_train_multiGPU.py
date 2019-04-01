import keras
from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data_process import DataProcess
from data_generator import DataGenerator
import Model as crnn_model
import parameter as params
import os
from keras.utils import multi_gpu_model
from batch_test import epoch_eval


# K.set_learning_phase(0)

class MyCheckPoint(keras.callbacks.Callback):
    def __init__(self, model): 
        self.model_to_save = model 
    def on_epoch_end(self, epoch, logs=None): 
        self.model_to_save.save('/data/output/crnn_weights_20190106/CRNN_model_at_epoch_%d.h5' % epoch)


def data_preprocess():
    # check train_data
    if os.path.exists(os.path.exists(params.cut_img_save_path)) and \
            os.path.exists(params.json_train_path) and \
            os.path.exists(params.json_val_path) and \
            os.path.exists(params.generate_key_path):
        print("train_data have been build, pre process task over !")
        return
    else:
        print("train_data not exist, do processing...")

    data_proc = DataProcess()
    data_proc.data_preprocess()
    # data_proc.generate_key()
    # data_proc.random_get_val()


def find_latest_weights(weight_path):
    file_list = []
    for root, dirs, files in os.walk(weight_path):
        for file_name in files:
            # print("find file: ", file_name)
            if 'CRNN' in file_name:
                file_list.append(file_name)
    if len(file_list) == 0:
        print("weights file list is empty.")
        return None
    else:
        file_list.sort(reverse=True)
        # print("latest weights file: ", file_list[0])
        return file_list[0]


def load_train_and_val_data():

    # train data path
    json_train_path = params.json_train_path
    json_val_path = params.json_val_path
    cut_img_save_path = params.cut_img_save_path
    key_path = params.char_path

    # data generator
    train_data = DataGenerator(img_dirpath=cut_img_save_path,
                               json_path=json_train_path,
                               char_path=key_path,
                               img_w=params.img_w,
                               img_h=params.img_h,
                               batch_size=params.batch_size,
                               downsample_factor=params.downsample_factor,
                               max_text_len=params.max_text_len)
    train_data.build_data()
    train_sample_num = train_data.n

    val_data = DataGenerator(img_dirpath=cut_img_save_path,
                             json_path=json_val_path,
                             char_path=key_path,
                             img_w=params.img_w,
                             img_h=params.img_h,
                             batch_size=params.batch_size,
                             downsample_factor=params.downsample_factor,
                             max_text_len=params.max_text_len)
    val_data.build_data()
    val_sample_num = val_data.n
    return train_data.next_batch(), \
           val_data.next_batch(), \
           train_sample_num, \
           val_sample_num


def train_model():
    # calc num_classes
    key_f = open(params.char_path, 'r', encoding='utf-8')
    chars = key_f.read()
    key_f.close()
    params.num_classes = len(chars) + 1
    print('params.num_classes: ', params.num_classes)

    template_model = crnn_model.get_Model(training=True)

    try:
        latest_weights = params.load_weights_path
        print("find latest_weights exists.", latest_weights)
        if latest_weights != None:
            template_model.load_weights(latest_weights)
            print("...load exist weights: ", latest_weights)
        else:
            print("history weights file not exist, train a new one.")
    except Exception as e:
        print('warn: ', str(e))
        print("historical weights data can not be used, train a new one...")
        pass

    model = multi_gpu_model(template_model, gpus=params.gpu_nums_in_multi_model)
    
    model.layers[-2].set_weights(template_model.get_weights())

    ada = Adadelta()

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=6,
                               mode='min',
                               verbose=1)
                                 
    my_checkpoint = MyCheckPoint(template_model)    
    
    tensor_board = TensorBoard(log_dir='/data/output')

    train_data_gen, val_data_gen, train_sample_num, val_sample_num = load_train_and_val_data()

    batch_size = params.batch_size
    epoch_num = params.epoch_num
    val_batch_size = params.val_batch_size

    for ep_i in range(epoch_num):
        print("epoch: ", ep_i+1)
        model.fit_generator(generator=train_data_gen,
                            steps_per_epoch=train_sample_num // batch_size,
                            epochs=ep_i + 1,
                            callbacks=[],
                            verbose=2,
                            initial_epoch=ep_i,
                            validation_data=val_data_gen,
                            validation_steps=val_sample_num // val_batch_size)

        curr_weights_path = "/data/output/crnn_weights_20190106/crnn_weights_v1.13_ep_%d.h5" % (ep_i + 1)
        template_model.save_weights(curr_weights_path)
        train_data_acc = epoch_eval.eval_on_generating_data(curr_weights_path)
        print(" -- train_data_acc: ", train_data_acc)
        real_data_acc, detail_info = epoch_eval.eval_on_real_data(curr_weights_path)
        print(" -- real_data_acc: ", real_data_acc, detail_info)

if __name__ == "__main__":
    data_preprocess()
    train_model()
