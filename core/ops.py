import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import warnings

def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )

  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar

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

def conv_sn(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", spectral_normed=False, update_collection=None, with_w=False, padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_dim
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if spectral_normed:
      conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                          strides=[1, d_h, d_w, 1], padding=padding)
    else:
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv

def linear_sn(input_, output_size, name="linear", spectral_normed=False, update_collection=None, stddev=None, bias_start=0.0, with_biases=True,
           with_w=False):
  shape = input_.get_shape().as_list()

  if stddev is None:
    stddev = np.sqrt(1. / (shape[1]))
  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                             tf.truncated_normal_initializer(stddev=stddev))
    if with_biases:
      bias = tf.get_variable("b", [output_size],
                             initializer=tf.constant_initializer(bias_start))
    if spectral_normed:
      mul = tf.matmul(input_, spectral_normed_weight(weight, update_collection=update_collection))
    else:
      mul = tf.matmul(input_, weight)
    if with_w:
      if with_biases:
        return mul + bias, weight, bias
      else:
        return mul, weight, None
    else:
      if with_biases:
        return mul + bias
      else:
        return mul