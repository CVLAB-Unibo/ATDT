import tensorflow as tf
import numpy as np

def cross_entropy_loss(logits,labels,num_classes):
    labels = tf.squeeze(labels, axis=3)
    epsilon = tf.constant(value=1e-10)
    mask = tf.cast(tf.where(tf.greater_equal(labels ,tf.ones_like(labels)* num_classes), tf.zeros_like(labels), tf.ones_like(labels)),tf.float32)
    labels = tf.where(tf.greater_equal(labels ,tf.ones_like(labels)* num_classes), tf.zeros_like(labels), labels)
    labels = tf.cast(labels,tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy = tf.multiply(cross_entropy, mask)
    cross_entropy_sum = tf.reduce_sum(cross_entropy)
    mask_d = tf.reduce_sum(mask)+epsilon
    cross_entropy_mean = cross_entropy_sum/mask_d
    return cross_entropy_mean

def l1_loss(pred, gt, threshold=100, do_mask=True): ##depth in m, default th to 100m
    e=1e-10
    valid_map = tf.cast(tf.logical_not(tf.equal(gt,0)),tf.float32)
    if do_mask:
        mask = tf.where(tf.greater(gt,threshold),tf.zeros_like(gt),tf.ones_like(gt))
        gt_norm = tf.multiply(tf.divide(gt,threshold),mask) # rescale depth gt beetween 0 and 1 and masking. required pred depth beetween 0 and 1
        res = tf.reduce_sum(tf.multiply(tf.abs(gt_norm-pred),mask)*valid_map)/(tf.reduce_sum(mask*valid_map) + e)
    else:
        gt_norm = tf.divide(tf.where(tf.greater(gt,threshold),tf.ones_like(gt)*threshold,gt),threshold)
        abs_err = tf.abs(pred-gt_norm)
        res = tf.reduce_sum(abs_err*valid_map)/tf.reduce_sum(valid_map)
    return res

def cos_loss(pred,gt):
	mask = tf.to_float(tf.greater(tf.round(tf.sqrt(tf.maximum(tf.reduce_sum(tf.pow(gt,2),axis=-1), 1e-10))),0))
	norm_pred = tf.divide(pred,tf.sqrt(tf.maximum(tf.reduce_sum(tf.pow(pred,2),axis=-1,keepdims=True), 1e-10)))
	norm_gt = tf.divide(gt,tf.sqrt(tf.maximum(tf.reduce_sum(tf.pow(gt,2),axis=-1,keepdims=True), 1e-10)))
	loss = tf.reduce_sum(mask*(1-tf.reduce_sum(norm_pred*norm_gt,axis=-1)))/tf.reduce_sum(mask)
	return loss
