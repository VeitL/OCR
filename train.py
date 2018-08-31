import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from data_loader import DataLoader
from configs import cfgs
from models import * 
from dataset import *
from utils import *

def run_training():
    dl = DataLoader(cfgs.train_dir, batch_size=cfgs.batch_size, is_training=True, 
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
    
    total_loss = build_total_loss(labels, logits, seq_length)

    optimizer = tf.train.AdamOptimizer(learning_rate=cfgs.learning_rate)
    train_op = optimizer.minimize(total_loss)

    saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dl.reset_op())
        epoch_counter = 0
        ite = 0
        while True:
            try:
                _, loss_, logits_, labels_, shapes_, images_, seq_length_, width_ = sess.run(
                    [train_op, total_loss, logits, labels, shapes, images, seq_length,width])
                ite += 1
                if ite % cfgs.log_interval == 0:
                    print('Iteration %s, loss %f' %(ite, loss_))
            except tf.errors.OutOfRangeError:
                print('Finish epoch %d' %epoch_counter)
                epoch_counter += 1
                if epoch_counter >= cfgs.n_epochs:
                    # save model
                    saver.save(sess, "%s/checkpoint-final.ckpt" %(cfgs.checkpoint_basedir))
                    print('Final checkpoint saved')
                    break
                if epoch_counter % cfgs.checkpoint_interval == 0:
                    saver.save(sess, "%s/checkpoint-%depochs.ckpt" %(cfgs.checkpoint_basedir, epoch_counter))
                    print('Checkpoint %d saved' %epoch_counter)
                sess.run(dl.reset_op())


if __name__ == '__main__':
    run_training()

