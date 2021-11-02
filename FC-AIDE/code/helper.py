import numpy as np
import tensorflow as tf

from code.nets.layers import Conv2d_filter

net = Conv2d_filter(1, filter_mode='d2')
tensor_helper = tf.ones((1, 24, 24, 1))
helper = net(tensor_helper)
