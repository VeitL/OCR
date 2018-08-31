import tensorflow as tf 
import scipy
import matplotlib.pyplot as plt
from data_loader import DataLoader
from configs import cfgs
from models import * 
from dataset import *
from utils import *

def run_demo():
    dl = DataLoader(cfgs.test_dir, batch_size=cfgs.batch_size, is_training=True, 
        num_devices=cfgs.num_devices)
    labels, shapes, images, width = dl.inputs()[0]
    char2id, id2char, num_labels = build_dictionary()
    num_labels = num_labels + 1 # to account for the blank label
    logits = fcn(images, num_labels, cfgs.recurrent_conv, cfgs.weight_decay, is_training=True)

    seq_length = tf.cast(width/4, tf.int32)
    preds,_ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, [1,0,2]), 
                                            seq_length,
                                            beam_width=128,
                                            top_paths=1,
                                            merge_repeated=True)

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dl.reset_op())
        # restore model
        checkpoint = cfgs.checkpoint
        if checkpoint == '':
            checkpoint = tf.train.latest_checkpoint(cfgs.checkpoint_basedir)
        saver.restore(sess, checkpoint)
        print('Model restored from %s' %checkpoint)

        while True:
            try:
                images_, preds_, width_ = sess.run([images, preds, width])
                preds_ = np.swapaxes(preds_, 0, 1)
                ids = preds_[1][0]
                ids[ids==0] = -1
                rows = preds_[0][0][:,0]
                cols = preds_[0][0][:,1]
                shape = tuple(preds_[2][0])
                tokens = scipy.sparse.coo_matrix((ids, (rows,cols)), shape=shape)
                tokens = np.array(tokens.todense())
                tokens[tokens==0] = -2
                tokens[tokens==-1] = 0
                string_preds = [token_ids_to_label(id2char, tokens[i,:]) for i in range(tokens.shape[0])]
                for i in range(tokens.shape[0]):
                    print(string_preds[i])
                    img = images_[i]
                    img = img[:,:width_[i],:]
                    img += 0.5
                    plt.imshow(img)
                    plt.text(40, int(width_[i]/2), string_preds[i])
                    plt.show()
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    run_demo()

