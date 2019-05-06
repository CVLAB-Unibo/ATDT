import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from core.ops import *
from core.models import *
from core.input import * 
from core.losses import *
from utils.utils import *

def factory(inputs, params,reuse_variables=False):
    if params.task=='semantic': return SemanticNetwork(inputs, params,reuse_variables)
    elif params.task == 'depth':  return DepthNetwork(inputs,params,reuse_variables)
    elif params.task == 'normals':  return NormalNetwork(inputs,params,reuse_variables)
    else: raise NotImplementedError("Please Implement this Task Network")

class Network(object):
    def __init__(self,inputs, params, reuse_variables=False):
        self.inputs=inputs
        self.params = params
        self.reuse_variables = reuse_variables

        if self.params.mode == 'train':
            self.images, self.labels = self.inputs
            training=True
        else: 
            self.images = self.inputs
            training=False

        if self.params.normalizer_fn == 'batch_norm':
            self.normalizer_fn = lambda x : tf.layers.batch_normalization(x,training=training)
        elif self.params.normalizer_fn == 'group_norm':
            self.normalizer_fn = lambda x : group_norm(x)
        elif self.params.normalizer_fn == 'instance_norm':
            self.normalizer_fn = lambda x : instance_norm(x)
        else: 
            self.normalizer_fn=None
        
        self.summary_images=[tf.summary.image("image",self.images)]
        self.summary_scalar=[]

        self.build()
        if self.params.mode == 'train':
            self.build_summary()

    def build(self):
        raise NotImplementedError("Please Implement this method")
    def build_summary(self):
        raise NotImplementedError("Please Implement this method")

class SemanticNetwork(Network):
    def __init__(self,inputs, params, reuse_variables=False):
        super().__init__(inputs, params,reuse_variables)
        
    def build(self):
        self.logits, self.features = build_model(self.images, self.params.num_classes, use_skips=self.params.use_skips, encoder=self.params.encoder,normalizer_fn=self.normalizer_fn)
        if self.params.mode == 'train': 
            self.loss = cross_entropy_loss(self.logits,self.labels,self.params.num_classes)
        self.pred_map = tf.expand_dims(tf.argmax(self.logits, axis=-1),axis=-1)
        self.pred = color_tensorflow(self.pred_map)
    
    def build_summary(self):
        gts_sum = color_tensorflow(self.labels)
        self.summary_images.append(tf.summary.image("pred",self.pred))
        self.summary_images.append(tf.summary.image("gt",gts_sum))
        if self.params.mode == 'train': 
            self.summary_scalar.append(tf.summary.scalar("loss", self.loss))

class DepthNetwork(Network):
    def __init__(self,inputs, params, reuse_variables=False):
        super().__init__(inputs, params,reuse_variables)

    def build(self):
        self.pred_map, self.features = build_model(self.images, 1, use_skips=self.params.use_skips, encoder=self.params.encoder,normalizer_fn=self.normalizer_fn)
        if self.params.mode == 'train':
            self.loss = l1_loss(self.pred_map,self.labels)
        self.pred = colormap_depth(tf.clip_by_value(self.pred_map,0,1), cmap='jet')
    
    def build_summary(self):
        gts_sum = colormap_depth(tf.clip_by_value(self.labels,0,100),cmap='jet')
        self.summary_images.append(tf.summary.image("pred",self.pred))
        self.summary_images.append(tf.summary.image("gt",gts_sum))
        if self.params.mode == 'train': 
            self.summary_scalar.append(tf.summary.scalar("loss", self.loss))        
    
class NormalNetwork(Network):
    def __init__(self,inputs, params, reuse_variables=False):
        super().__init__(inputs, params,reuse_variables)

    def build(self):
        self.pred_map, self.features = build_model(self.images, 3, use_skips=self.params.use_skips, encoder=self.params.encoder,normalizer_fn=self.normalizer_fn)
        self.pred_map = tf.nn.tanh(self.pred_map)
        if self.params.mode == 'train':
            self.loss = cos_loss(self.pred_map,self.labels)
        self.pred = (self.pred_map+1)/2*255
    
    def build_summary(self):
        gts_sum = tf.cast(tf.clip_by_value((self.labels+1)/2*255,0,255),tf.uint8)
        self.summary_images.append(tf.summary.image("pred",self.pred))
        self.summary_images.append(tf.summary.image("gt",gts_sum))
        if self.params.mode == 'train': 
            self.summary_scalar.append(tf.summary.scalar("loss", self.loss))

class TransferNetwork(Network):
    def __init__(self,inputs, params, model='dilated-resnet', encoder_source=True, encoder_target=True, decoder_target=True, reuse_variables=False, feature_level=-1):
        self.encoder_source = encoder_source
        self.encoder_target = encoder_target
        self.decoder_target = decoder_target
        self.target_task = params.task
        self.model=model
        self.feature_level=feature_level

        super().__init__(inputs, params,reuse_variables)

    def build_encoder(self):
        with tf.variable_scope('model'):
            if self.model == 'vgg':
                print("Building VGG Encoder")
                features, skips = build_vgg(inputs, self.params.use_skips, normalizer_fn=self.normalizer_fn, reuse_variables=self.reuse_variables)
            elif self.model == 'resnet':
                print("Building ResNet50 Encoder")
                features, skips = build_resnet50(inputs, self.params.use_skips, normalizer_fn=self.normalizer_fn, reuse_variables=self.reuse_variables)
            elif self.model == 'dilated-resnet':
                print("Building Dilated-Resnet Encoder")
                features, skips = build_dilated_resnet50(inputs, self.params.use_skips, normalizer_fn=self.normalizer_fn, reuse_variables=self.reuse_variables)
            return features, skips

    def build_decoder(self,features):
        if self.target_task == 'semantic':
            ch=self.params.num_classes
        elif self.target_task == 'depth':
            ch=1
        else:
            ch=3
        with tf.variable_scope('model'):
            if self.model == 'vgg':
                print("Building VGG Decoder")
                output = build_decoder_vgg(features, ch, normalizer_fn=self.normalizer_fn, reuse_variables=self.reuse_variables)
            elif self.model == 'resnet':
                print("Building ResNet50 Decoder")
                output = build_decoder_resnet(features, ch, normalizer_fn=self.normalizer_fn, reuse_variables=self.reuse_variables)
            elif self.model == 'dilated-resnet':
                print("Building Dilated-Resnet Decoder")
                output = build_decoder_dilated_resnet(features, ch, normalizer_fn=self.normalizer_fn, reuse_variables=self.reuse_variables)
        return output

    def build(self):
        #### ENCODERS ####
        if self.encoder_source:
            with tf.variable_scope('source'):
                self.features_source, skips = self.build_encoder(inputs,args.encoder,args.use_skips)
                self.features_source = features_source[self.feature_level]
        if self.encoder_source:
            with tf.variable_scope('target'):
                self.features_target, skips = self.build_encoder(inputs,args.encoder,args.use_skips)
                self.features_target = features_target[self.feature_level]

        #### TRANSFER ####
        with tf.variable_scope('transfer',reuse=self.reuse_variables):
            print("Building Transfer Network")
            self.adapted_features=transfer_network(self.features_source)

        #### DECODERS ####
        if self.decoder_target:
            self.pred_map = build_decoder(self.adapted_feature)
        else:
            self.pred_map = self.adapted_feature
        
        if self.params.mode == 'train':
            self.loss=tf.reduce_mean(tf.pow(self.features_target-self.adapted_feature,2))

    def build_summary(self):
        if self.decoder_target:
            if self.target_task == 'semantic':
                output_sum = color_tensorflow(tf.expand_dims(tf.cast(tf.argmax(self.pred_map,axis=-1),tf.uint8),axis=-1))
            elif self.target_task == 'depth':
                output_sum = colormap_depth(tf.clip_by_value(self.pred_map,0,1), cmap='jet')
            elif self.target_task == 'normals':
                output_sum = (tf.nn.tanh(self.pred_map)+1)/2*255
            self.summary_images.append(tf.summary.image("pred", output_sum ,max_outputs=1))
        
        if self.params.mode == 'train': 
            self.summary_scalar.append(tf.summary.scalar("loss", self.loss))