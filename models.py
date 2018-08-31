import tensorflow as tf 

def fcn(images, num_labels, recurrent, weight_decay, is_training):
    # input image: H x W x C
    T = 1
    init = tf.glorot_normal_initializer()
    reg = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    with tf.variable_scope('block1'):
        net = tf.layers.conv2d(images, filters=32, kernel_size=[3,3], padding='same', name='conv1_1', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.conv2d(net, filters=32, kernel_size=[3,3], padding='same', name='conv1_2', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='same')
        # intemediate shape: H/2 x W/2
    with tf.variable_scope('block2'):
        net = tf.layers.conv2d(net, filters=64, kernel_size=[3, 3], padding='same', name='conv2_1', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.batch_normalization(net, momentum=0.9, training=is_training)
        net = tf.layers.conv2d(net, filters=64, kernel_size=[3, 3], padding='same', name='conv2_2', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        if recurrent:
            for _ in range(T):
                net = tf.layers.conv2d(net, filters=64, kernel_size=[3, 3], padding='same', name='conv2_2', reuse=True)
                net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='same')
        # intemediate shape: H/4 x W/4
    with tf.variable_scope('block3'):
        net = tf.layers.conv2d(net, filters=128, kernel_size=[3, 3], padding='same', name='conv3_1', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.batch_normalization(net, momentum=0.9, training=is_training)
        net = tf.layers.conv2d(net, filters=128, kernel_size=[3, 3], padding='same', name='conv3_2', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        if recurrent:
            for _ in range(T):
                net = tf.layers.conv2d(net, filters=128, kernel_size=[3, 3], padding='same', name='conv3_2', reuse=True)
                net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 1], padding='same')
        # intemediate shape: H/8 x W/4
    with tf.variable_scope('block4'):
        net = tf.layers.conv2d(net, filters=256, kernel_size=[3, 3], padding='same', name='conv4_1', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.batch_normalization(net, momentum=0.9, training=is_training)
        net = tf.layers.conv2d(net, filters=256, kernel_size=[3, 3], padding='same', name='conv4_2', kernel_initializer=init, kernel_regularizer=reg)
        if recurrent:
            for _ in range(T):
                net = tf.layers.conv2d(net, filters=256, kernel_size=[3, 3], padding='same', name='conv4_2', reuse=True)
                net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 1], padding='same')
        # intemediate shape: H/16 x W/4
    with tf.variable_scope('block5'):
        net = tf.layers.conv2d(net, filters=512, kernel_size=[2, 3], strides=(2,1), padding='same', name='conv5_1', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.conv2d(net, filters=512, kernel_size=[1, 5], padding='same', name='conv5_2', kernel_initializer=init, kernel_regularizer=reg)
        net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.layers.conv2d(net, filters=num_labels, kernel_size=[1, 7], padding='same', name='conv5_3', kernel_initializer=init, kernel_regularizer=reg)
        # intemediate shape: H/32 x W/4
        # net = tf.nn.leaky_relu(net, alpha=0.1)
        net = tf.squeeze(net, axis=1)
    return net

def build_total_loss(labels, logits, sequence_length):
    ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=sequence_length,
        ignore_longer_outputs_than_inputs=True, time_major=False)
    reg_loss = tf.losses.get_regularization_loss()
    total_loss = tf.reduce_mean(ctc_loss) + reg_loss
    return total_loss


