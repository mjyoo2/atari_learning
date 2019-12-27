from stable_baselines.a2c.utils import conv, conv_to_fc
import tensorflow as tf
import numpy as np

def nature_cnn_ex2(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = conv(scaled_images, 'c1', n_filters=32, filter_size=8,  stride=4, init_scale=np.sqrt(2), **kwargs)
    layer_1 = activ(layer_1)
    layer_2 = conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs)
    layer_2 = activ(tf.layers.batch_normalization(layer_2))
    layer_2 = conv(layer_2, 'c2_1', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = activ(tf.layers.batch_normalization(layer_2))
    layer_2 = conv(layer_2, 'c2_2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_2 = activ(tf.layers.batch_normalization(layer_2))
    layer_2 = tf.nn.max_pool(layer_2, (3, 3), (2, 2), "VALID")
    layer_3 = conv(layer_2, 'c31', n_filters=32, filter_size=1, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_3 = activ(tf.layers.batch_normalization(layer_3))
    layer_3 = conv(layer_3, 'c3', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_3 = activ(tf.layers.batch_normalization(layer_3))
    layer_4 = tf.nn.avg_pool(layer_3, (4, 4), (1, 1), 'VALID')
    layer_4 = conv_to_fc(layer_4)
    return layer_4