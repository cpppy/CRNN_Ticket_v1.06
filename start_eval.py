from batch_test import epoch_eval
import os


if __name__=='__main__':
    print('start doing batch_testing...')
    weights_dir = '/data/output/crnn_ticket_20190327'

    #weights_fn_list = ['crnn_weights_d10w_20190325_effects_ep_3.h5']
    weights_fn_list = os.listdir(weights_dir)
    weights_fn_list = sorted(weights_fn_list)
    # weights_fn_list = ['crnn_weights_d10w_ep_%d.h5'%i for i in range(1,20)]
    #weights_fn_list = weights_fn_list[:1]
    for idx, weights_fn in enumerate(weights_fn_list):
        print('--------------------------------------')
        print(idx, weights_fn)
        weights_path = os.path.join(weights_dir, weights_fn)
        #acc = epoch_eval.eval_on_generating_data(weights_path)
        #print('acc', acc)
        acc, test_res = epoch_eval.eval_on_real_data(weights_path)
        print('test_res: ', test_res)
