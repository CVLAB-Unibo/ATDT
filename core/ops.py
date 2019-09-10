import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import warnings

def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu, normalizer_fn=None, padding='SAME'):
    if padding == 'SAME':
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        output=slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)
    elif padding == 'VALID':
        output=slim.conv2d(x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)
    return output

def conv_block(x, num_out_layers, kernel_size,normalizer_fn=None):
    conv1 = conv(x,     num_out_layers, kernel_size, 1, normalizer_fn=normalizer_fn)
    conv2 = conv(conv1, num_out_layers, kernel_size, 2, normalizer_fn=normalizer_fn)
    return conv2

def conv_dilated(x, num_out_layers, kernel_size, stride=1, rate=1,activation_fn=tf.nn.elu, normalizer_fn=None):
    return slim.conv2d(x,num_out_layers,kernel_size,stride,'SAME',rate=rate,activation_fn=activation_fn, normalizer_fn=normalizer_fn)

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def resconv(x, num_layers, stride, normalizer_fn=None):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1, normalizer_fn=normalizer_fn)
    conv2 = conv(conv1,     num_layers, 3, stride, normalizer_fn=normalizer_fn)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None, normalizer_fn=normalizer_fn)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None, normalizer_fn=normalizer_fn)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def resblock_generator(x, num_layers, activation=tf.nn.elu, normalizer_fn=None):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    conv1 = conv(x,num_layers,3,1,tf.nn.elu, normalizer_fn=normalizer_fn)
    conv2 = conv(x,num_layers,3,1,None, normalizer_fn=normalizer_fn)
    if do_proj:
        shortcut = conv(x, num_layers, 3, 1, None, normalizer_fn=normalizer_fn)
    else:
        shortcut = x
    if activation:
        return activation(conv2 + shortcut)
    else:
        return conv2 + shortcut

def resconv_dilated(x, num_layers, rate, normalizer_fn=None):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1, normalizer_fn=normalizer_fn)
    conv2 = conv_dilated(conv1, num_layers, 3, rate=rate, normalizer_fn=normalizer_fn)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None, normalizer_fn=normalizer_fn)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, 1, None, normalizer_fn=normalizer_fn)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks, normalizer_fn=None):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1, normalizer_fn=normalizer_fn)
    out = resconv(out, num_layers, 2, normalizer_fn=normalizer_fn)
    return out

def resblock_dilated(x, num_layers, num_blocks, rate, normalizer_fn=None):
    out = x
    for i in range(num_blocks - 1):
        out = resconv_dilated(out, num_layers, rate, normalizer_fn=normalizer_fn)
    out = resconv_dilated(out, num_layers, rate, normalizer_fn=normalizer_fn)
    return out

def upconv(x, num_out_layers, kernel_size, scale, normalizer_fn=None):
    upsample = upsample_nn(x, scale)
    out = conv(upsample, num_out_layers, kernel_size, 1, normalizer_fn=normalizer_fn)
    return out

def deconv(x, num_out_layers, kernel_size, scale,normalizer_fn=None):
    p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    out = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
    return out[:,3:-1,3:-1,:]

def instance_norm(input):
    depth = input.get_shape()[3]
    scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def group_norm(x, G= 32, eps=1e-5):
    # x: input features with shape [N,C,H,W] 
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, H, W, C = x.get_shape()

    gamma=tf.get_variable("gamma", [C], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    beta= tf.get_variable("beta", [C], initializer=tf.constant_initializer(0.0))
    
    x = tf.reshape(x, [N, H, W, C // G, G])
    mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, H, W, C])
    return x * gamma + beta