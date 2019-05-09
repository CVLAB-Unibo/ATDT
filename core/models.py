import numpy as np
import tensorflow as tf
from collections import namedtuple
from core.ops import *
from utils.utils import *

def SNDCGAN_Discrminator(x, batch_size=16,hidden_activation=tf.nn.leaky_relu, output_dim=1, scope='critic', update_collection=tf.GraphKeys.UPDATE_OPS):
    with tf.variable_scope(scope):
        c0_0 = hidden_activation(conv_sn(   x,  64, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_0'))
        c0_1 = hidden_activation(conv_sn(c0_0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_1'))
        c1_0 = hidden_activation(conv_sn(c0_1, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_0'))
        c1_1 = hidden_activation(conv_sn(c1_0, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_1'))
        c2_0 = hidden_activation(conv_sn(c1_1, 256, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_0'))
        c2_1 = hidden_activation(conv_sn(c2_0, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_1'))
        c3_0 = hidden_activation(conv_sn(c2_1, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3_0'))
        c3_0 = tf.reshape(c3_0, [batch_size, -1])
        l4 = linear_sn(c3_0, output_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
    return tf.reshape(l4, [-1])


def build_generator(inputs, reuse_variables=False, normalizer_fn=None):
    with tf.variable_scope('generator',reuse=reuse_variables):
        convs = []
        convs.append(resblock_generator(inputs,      2048))
        convs.append(resblock_generator(convs[-1],      512, normalizer_fn=normalizer_fn))
        convs.append(resblock_generator(convs[-1],      128, normalizer_fn=normalizer_fn))
        convs.append(resblock_generator(convs[-1],inputs.shape[-1].value))
        return convs[-1], convs

def build_discriminator(inputs, reuse_variables=False, normalizer_fn=None):
    with tf.variable_scope('discriminator',reuse=reuse_variables):
        convs = []
        convs.append(conv(inputs,1024,3,2,normalizer_fn=normalizer_fn))
        convs.append(conv(convs[-1],512,3,2,normalizer_fn=normalizer_fn))
        convs.append(conv(convs[-1],1,3,1,activation_fn=None,normalizer_fn=normalizer_fn))
        return convs,convs

def build_vgg(inputs, use_skips = False, reuse_variables=False, normalizer_fn=None):
    with tf.variable_scope('encoder', reuse=reuse_variables):
        convs = []
        convs.append(conv_block(inputs,  32, 7, normalizer_fn=normalizer_fn)) # H/2
        convs.append(conv_block(convs[-1],             64, 5, normalizer_fn=normalizer_fn)) # H/4
        convs.append(conv_block(convs[-1],            128, 3, normalizer_fn=normalizer_fn)) # H/8
        convs.append(conv_block(convs[-1],            256, 3, normalizer_fn=normalizer_fn)) # H/16
        convs.append(conv_block(convs[-1],            512, 3, normalizer_fn=normalizer_fn)) # H/32
        convs.append(conv_block(convs[-1],            512, 3, normalizer_fn=normalizer_fn)) # H/64
        convs.append(conv_block(convs[-1],            512, 3, normalizer_fn=normalizer_fn)) # H/128
        #skips
        skips=[]
        if use_skips:
            print("Adding Skip Connections")
            skips=convs[:-1]

        return convs,skips

def build_resnet50(inputs, use_skips = False, reuse_variables=False, normalizer_fn=None):
    with tf.variable_scope('encoder', reuse=reuse_variables):
        convs = []
        convs.append(    conv(inputs,     64, 7, 2, normalizer_fn=normalizer_fn)) # H/2  -   64D
        convs.append( maxpool(convs[-1],          3)) # H/4  -   64D
        convs.append(resblock(convs[-1],      64, 3, normalizer_fn=normalizer_fn)) # H/8  -  256D
        convs.append(resblock(convs[-1],     128, 4, normalizer_fn=normalizer_fn)) # H/16 -  512D
        convs.append(resblock(convs[-1],     256, 6, normalizer_fn=normalizer_fn)) # H/32 - 1024D
        convs.append(resblock(convs[-1],     512, 3, normalizer_fn=normalizer_fn)) # H/64 - 2048D
    
        #skips
        skips=[]
        if use_skips:
            print("Adding Skip Connections")
            skips=convs[:-1]

        return convs, skips

def build_dilated_resnet50(inputs, use_skips = False, reuse_variables=False, normalizer_fn=None):
    with tf.variable_scope('encoder', reuse=reuse_variables):

        convs = []
        convs.append(    conv(inputs,     64, 7, 2, normalizer_fn=normalizer_fn)) # H/2  -   64D
        convs.append( maxpool(convs[-1],          3)) # H/4  -   64D
        convs.append(resblock(convs[-1],      64, 3, normalizer_fn=normalizer_fn)) # H/8  -  256D
        convs.append(resblock(convs[-1],     128, 4, normalizer_fn=normalizer_fn)) # H/16 -  512D
        convs.append(resblock_dilated(convs[-1],     256, 6, 2, normalizer_fn=normalizer_fn)) # H/16 - 1024D - R2
        convs.append(resblock_dilated(convs[-1],     512, 3, 4, normalizer_fn=normalizer_fn)) # H/16 - 2048D - R4
        
        #skips
        skips=[]
        if use_skips:
            print("Adding Skip Connections")
            skips=convs[1:-2]
        
        return convs[1:], skips

def build_decoder_vgg(inputs, out_channels, skips= [], reuse_variables=False, normalizer_fn=None):
    print("Adding Skip Connections") if skips else None
    with tf.variable_scope('decoder', reuse=reuse_variables):
        upconv7 = upconv(inputs,  512, 3, 2, normalizer_fn=normalizer_fn) #H/64
        if skips:
            upconv7 = tf.concat([upconv7, skips[5]], 3)
        iconv7  = conv(upconv7,  512, 3, 1, normalizer_fn=normalizer_fn)
        upconv6 = upconv(iconv7, 512, 3, 2, normalizer_fn=normalizer_fn) #H/32
        if skips:
            upconv6 = tf.concat([upconv6, skips[4]], 3)
        iconv6  = conv(upconv6,  512, 3, normalizer_fn=normalizer_fn)
        upconv5 = upconv(iconv6, 256, 3, 2, normalizer_fn=normalizer_fn) #H/16
        if skips:
            upconv5 = tf.concat([upconv5, skips[3]], 3)
        iconv5  = conv(upconv5,  256, 3, 1, normalizer_fn=normalizer_fn)
        upconv4 = upconv(iconv5, 128, 3, 2, normalizer_fn=normalizer_fn) #H/8
        if skips:
            upconv4 = tf.concat([upconv4, skips[2]], 3)
        iconv4  = conv(upconv4,  128, 3, 1, normalizer_fn=normalizer_fn)
        upconv3 = upconv(inputs,  64, 3, 2, normalizer_fn=normalizer_fn) #H/4
        if skips:
            upconv3 = tf.concat([upconv3, skips[1]], 3)
        iconv3  = conv(upconv3,   64, 3, 1, normalizer_fn=normalizer_fn)
        upconv2 = upconv(iconv3,  32, 3, 2, normalizer_fn=normalizer_fn) #H/2
        if skips:
            upconv2 = tf.concat([upconv2, skips[0]], 3)
        iconv2  = conv(upconv2,   32, 3, 1, normalizer_fn=normalizer_fn)
        upconv1 = upconv(iconv2,  out_channels, 3, 2, normalizer_fn=None) #H
        iconv1  = conv(upconv1,   out_channels, 3, 1, activation_fn=None)
        return iconv1

def build_decoder_resnet(inputs, out_channels, skips= [], reuse_variables=False, normalizer_fn=None):
    print("Adding Skip Connections") if skips else None        
    with tf.variable_scope('decoder', reuse=reuse_variables):
        upconv6 = upconv(inputs,   512, 3, 2, normalizer_fn=normalizer_fn) #H/32
        if skips:
            upconv6 = tf.concat([upconv6, skips[4]], 3)
        iconv6  = conv(upconv6,   512, 3, 1, normalizer_fn=normalizer_fn)
        upconv5 = upconv(iconv6, 256, 3, 2, normalizer_fn=normalizer_fn) #H/16
        if skips:
            upconv5 = tf.concat([upconv5, skips[3]], 3)
        iconv5  = conv(upconv5,   256, 3, 1, normalizer_fn=normalizer_fn)
        upconv4 = upconv(iconv5,  128, 3, 2, normalizer_fn=normalizer_fn) #H/8
        if skips:
            upconv4 = tf.concat([upconv4, skips[2]], 3)
        iconv4  = conv(upconv4,   128, 3, 1, normalizer_fn=normalizer_fn)
        upconv3 = upconv(iconv4,   64, 3, 2, normalizer_fn=normalizer_fn) #H/4
        if skips:
            upconv3 = tf.concat([upconv3, skips[1]], 3)
        iconv3  = conv(upconv3,    64, 3, 1, normalizer_fn=normalizer_fn)
        upconv2 = upconv(iconv3,   32, 3, 2) #H/2
        if skips:
            upconv2 = tf.concat([upconv2, skips[0]], 3)
        iconv2  = conv(upconv2,    32, 3, 1, normalizer_fn=normalizer_fn)

        upconv1 = upconv(iconv2,  out_channels, 3, 2, normalizer_fn=None) #H
        iconv1  = conv(upconv1,   out_channels, 3, 1, activation_fn=None)
        return iconv1

def build_decoder_dilated_resnet(inputs, out_channels, skips= [], reuse_variables=False, normalizer_fn=None):
    print("Adding Skip Connections") if skips else None
    with tf.variable_scope('decoder', reuse=reuse_variables):
        upconv4 = upconv(inputs,  128, 3, 2, normalizer_fn=normalizer_fn) #H/8
        if skips:
            upconv4 = tf.concat([upconv4, skips[-1]], 3)
        iconv4  = conv(upconv4,   128, 3, 1, normalizer_fn=normalizer_fn)
        upconv3 = upconv(iconv4,   64, 3, 2, normalizer_fn=normalizer_fn) #H/4
        if skips:
            upconv3 = tf.concat([upconv3, skips[-2]], 3)
        iconv3  = conv(upconv3,    64, 3, 1, normalizer_fn=normalizer_fn)
        upconv2 = upconv(iconv3,   32, 3, 2) #H/2
        if skips:
            upconv2 = tf.concat([upconv2, skips[-3]], 3)
        iconv2  = conv(upconv2,    32, 3, 1, normalizer_fn=normalizer_fn)
        upconv1 = upconv(iconv2,  out_channels, 3, 2, normalizer_fn=None) #H
        iconv1  = conv(upconv1,   out_channels, 3, 1, activation_fn=None)
        return iconv1

def build_model(inputs, out_channels, use_skips=False, encoder='dilated-resnet', name='model', reuse_variables=False, normalizer_fn=None):
    with tf.variable_scope(name, reuse=reuse_variables):
        if encoder == 'vgg':
            print("Building VGG")
            features, skips = build_vgg(inputs, use_skips=use_skips, normalizer_fn=normalizer_fn)
            output = build_decoder_vgg(features[-1], out_channels, skips, normalizer_fn=normalizer_fn)
            return output, features
        elif encoder == 'resnet':
            print("Building ResNet50")
            features, skips = build_resnet50(inputs, use_skips=use_skips, normalizer_fn=normalizer_fn)
            output = build_decoder_resnet(features[-1], out_channels, skips, normalizer_fn=normalizer_fn)
            return output, features
        elif encoder == 'dilated-resnet':
            print("Building Dilated-ResNet50")
            features, skips = build_dilated_resnet50(inputs, use_skips=use_skips, normalizer_fn=normalizer_fn)
            output = build_decoder_dilated_resnet(features[-1], out_channels, skips, normalizer_fn=normalizer_fn)
            return output, features
        else:
            raise NotImplementedError("Architecture not implemented")

def transfer_network(features_source, normalizer_fn=None):
    conv1 = conv(features_source,2048,3,2)
    conv2 = conv(conv1,2048,3,2) 
    upconv1 = upconv(conv2,  2048, 3, 2)
    iconv1  = conv(upconv1,   2048, 3, 1)
    upconv2 = upconv(iconv1,  2048, 3, 2)
    adapted_feature  = conv(upconv2,   features_source.get_shape()[-1], 3, 1)
    return adapted_feature