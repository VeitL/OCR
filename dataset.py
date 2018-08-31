# build TFRecord dataset
import tensorflow as tf 
import numpy as np
import glob, os
from PIL import Image
import math 
from utils import *

MAX_NUM_SUBFOLDERS = 10
NUM_IMAGES = 1000
MAX_HEIGHT = 32
MAX_WIDTH = 256

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    image = Image.open(filename)
    image = np.array(image)
    shape = np.array(image.shape, np.int32)
    to_write = (len(image.shape) > 1 and shape[0] > 20 
                and shape[1] > 20 and shape[1] <= MAX_WIDTH 
                and shape[0] <= MAX_HEIGHT )
    width = 0
    if to_write:
        width = shape[1]
        new_im = np.zeros((MAX_HEIGHT, MAX_WIDTH, shape[2]), dtype=image.dtype)
        new_im[:shape[0],:shape[1],:] = image
        image = new_im
        shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes(), width, to_write # convert image to raw data bytes in the array.

def write_to_tfrecord(writer, label_token, shape, width, binary_image):
    example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label_token),
                'shape': _bytes_feature(shape),
                'image': _bytes_feature(binary_image),
                'width': _int64_feature([width]),                
                }))
    writer.write(example.SerializeToString())

def write_tfrecord(image_file, char2id, writer):
    shape, binary_image, width, to_write = get_image_binary(image_file)
    if to_write:
        label = image_file.strip().split('/')[-1].split('_')[1]
        label_token = label_to_token_ids(char2id, label)
        if len(label_token) < math.ceil(width / 4):
            write_to_tfrecord(writer, label_token, shape, width, binary_image)
            return True
    return False

def write_shard(image_files, char2id, fname):
    writer = tf.python_io.TFRecordWriter(fname)
    for image_file in image_files:
        shape, binary_image, width, to_write = get_image_binary(image_file.strip())
        if to_write:
            label = image_file.strip().split('/')[-1].split('_')[1]
            label_token = label_to_token_ids(char2id, label)
            if len(label_token) < math.ceil(width / 4):
                write_to_tfrecord(writer, label_token, shape, width, binary_image)
    writer.close()


def create_dataset_synth_word(dir, save_fname):
    writer = tf.python_io.TFRecordWriter(save_fname)
    folders = glob.glob('%s/*' %dir)
    n_folders = len(folders)
    img_counter = 0
    char2id, ind2char, _ = build_dictionary()
    for i in range(n_folders):
        # process one folder (each has a number of subfolders)
        subfolders = glob.glob('%s/*' %folders[i])
        for sf in subfolders:
            # each subfolder contains hunderds of images (with labels on the filename)
            filenames = glob.glob('%s/*.jpg' %sf)
            for fname in filenames:
                flag = write_tfrecord(fname, char2id, writer)
                if not flag:
                    continue
                img_counter += 1
                if NUM_IMAGES > 0 and img_counter >= NUM_IMAGES:
                    writer.close()
                    return
                if img_counter % 10 == 0:
                    print(img_counter)
    writer.close()

def split_train_val_test(base_dir):
    folders = glob.glob('%s/*' %base_dir)
    n_folders = len(folders)
    all_subfolders = list([])
    n_subfolders = 0
    for i in range(n_folders):
        subfolders = glob.glob('%s/*' %folders[i])
        all_subfolders.extend(subfolders)
        n_subfolders += len(subfolders)
        if MAX_NUM_SUBFOLDERS > 0 and n_subfolders >= MAX_NUM_SUBFOLDERS:
            break
    train_cutoff = int(n_subfolders * 0.7)
    val_cutoff = int(n_subfolders * 0.8)
    train_subfolders = [all_subfolders[i] for i in range(train_cutoff)]
    val_subfolders = [all_subfolders[i] for i in range(train_cutoff, val_cutoff)]
    test_subfolders = [all_subfolders[i] for i in range(val_cutoff, n_subfolders)]
    return train_subfolders, val_subfolders, test_subfolders

def get_img_list(img_folders):
    with open('tmp.txt', 'w') as fout:
        for folder in img_folders:
            fnames = glob.glob('%s/*.jpg' %folder)
            for fname in fnames:
                fout.write('%s\n' %fname)
    with open('tmp.txt', 'r') as fin:
        img_list = fin.readlines()
        return img_list


def create_tf_records(img_folders, save_dir, n_shards=5):
    char2id, id2char, _ = build_dictionary()
    img_list = get_img_list(img_folders)
    num_digits = math.ceil( math.log10( n_shards - 1 ))
    shard_format = '%0'+ ('%d'%num_digits) + 'd'
    images_per_shard = int(math.ceil( len(img_list) / float(n_shards) ))
    start_shard = 0
    for i in range(start_shard,n_shards):
        start = i*images_per_shard
        end   = (i+1)*images_per_shard
        out_filename = save_dir+'-'+(shard_format % i)+'.tfrecord'
        print(str(i),'of',str(n_shards),'[',str(start),':',str(end),']',out_filename)
        write_shard(img_list[start:end], char2id, out_filename)
    # Clean up writing last shard
    start = n_shards*images_per_shard
    out_filename = save_dir+'-'+(shard_format % n_shards)+'.tfrecord'
    print(str(i),'of',str(n_shards),'[',str(start),':]',out_filename)


if __name__ == "__main__":
    base_dir = '~/Documents/mjsynth/mnt/ramdisk/max/90kDICT32px'
    # save_fname = './data/synth_word.tfrecords'
    # create_dataset_synth_word(base_dir, save_fname)
    train_sfs, val_sfs, test_sfs = split_train_val_test(base_dir)
    create_tf_records(train_sfs, './data/train/mjsynth', n_shards=5)
    create_tf_records(val_sfs, './data/val/mjsynth', n_shards=2)
    create_tf_records(test_sfs, './data/test/mjsynth', n_shards=2)
