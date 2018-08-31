import tensorflow as tf
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import glob
from utils import *

class DataLoader():
    """ Main Pipeline Class with some help functions"""

    def __init__(self, data_dir, batch_size=16, is_training=True, num_devices=1):
        tfrecord_files = glob.glob('%s/*.tfrecord' %data_dir)
        self.dataset = tf.data.TFRecordDataset(tfrecord_files)
        self.iterator = None
        self._batch_size = batch_size
        self._is_training = is_training
        self._num_devices = num_devices

    def _parse_example(self, example_proto):
        features={
            'label': tf.VarLenFeature(dtype=tf.int64),
            'shape': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        shape = tf.decode_raw(parsed_features['shape'], tf.int32)
        image = tf.reshape(image, shape)

        # convert to float32
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5) # normalization
        # first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
        # image = tf.concat([first_row, image], 0) # make the height = 32
        image.set_shape([32, None, 3])
        
        # get label
        label = tf.cast(parsed_features['label'], tf.int32)
        width = tf.cast(parsed_features['width'], tf.int32)
        return label, shape, image, width

    def build_dataset(self):
        dataset = self.dataset
        if self._is_training:
            dataset = dataset.shuffle(buffer_size=self._batch_size*5)
            # dataset = dataset.repeat(self._num_epochs)
        dataset = dataset.map(self._parse_example)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self.dataset = dataset
        self.iterator = self.dataset.make_initializable_iterator()

    def reset_op(self):
        if self.iterator is None:
            self.build_dataset()
        return self.iterator.initializer

    def inputs(self):
        if self.iterator is None:
            self.build_dataset()
        if self._num_devices is not None:
            result = []
            for d in range(self._num_devices):
                result.append(self.iterator.get_next())
            return result
        return self.iterator.get_next()
