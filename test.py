import tensorflow as tf 
import scipy
import matplotlib.pyplot as plt
from data_loader import DataLoader
from configs import cfgs
from models import * 
from dataset import *
from utils import *

def _get_metrics(predictions, labels):
    hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
    label_errors = tf.edit_distance(hypothesis, labels, normalize=False)
    sequence_errors = tf.cast((label_errors > 0), tf.float32)
    return label_errors, sequence_errors

def run_testing():
    dl = DataLoader(cfgs.test_dir, batch_size=cfgs.batch_size, is_training=True, 
        num_devices=cfgs.num_devices)
    labels, shapes, images, width = dl.inputs()[0]
    char2id, id2char, num_labels = build_dictionary()
    num_labels = num_labels + 1 # to account for the blank label
    logits = fcn(images, num_labels, cfgs.recurrent_conv, cfgs.weight_decay, is_training=True)

    seq_length = tf.cast(width/4, tf.int32)
    ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=seq_length,
        ignore_longer_outputs_than_inputs=True, time_major=False)
    loss = tf.reduce_mean(ctc_loss)

    preds, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, [1,0,2]), 
                                            seq_length,
                                            beam_width=128,
                                            top_paths=1,
                                            merge_repeated=True)
    label_error, sequence_error = _get_metrics(preds, labels)

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

        ite = 0
        losses = list([])
        label_errors = list([])
        sequence_errors = list([])
        while True:
            try:
                loss_, label_error_, sequence_error_, preds_ = sess.run([ctc_loss, label_error, sequence_error, preds])
                losses.extend(loss_) 
                label_errors.extend(label_error_) 
                sequence_errors.extend(sequence_error_)
                ite += 1 
                print('Testing iteration %d' %ite)
            except tf.errors.OutOfRangeError:
                # print test statistics
                print('----------------')                
                print('Average CTC loss: %f' %np.array(losses).mean())
                print('Average label error: %f' %np.array(label_errors).mean())
                print('Average sequence error: %f' %np.array(sequence_errors).mean())
                break


if __name__ == '__main__':
    run_testing()

