#!/usr/bin/env python

import os
import sys
import numpy as np
import tensorflow as tf
import datetime
import argparse
import cv2
import time

import core.NetworkFactory as NetworkFactory
from core.input import Dataloader
from utils.utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list_train', dest='input_list_train', default='input_list_train.txt', help='training relative path to each sample, gt separeted by ;')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

parser.add_argument('--steps', dest='steps', type=int, default=100000, help='# of steps')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--model', default='dilated-resnet', choices=['vgg','resnet','dilated-resnet'], help='resnet, dilated-resnet or vgg')
parser.add_argument('--task', dest='task', required=True, choices=['semantic','depth','normals'], help='task')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=11, help='# of classes')

parser.add_argument('--use_skips', dest='use_skips', action='store_true', help='use skip connection beetween encoder and decoder')
parser.set_defaults(use_skips=False)

parser.add_argument('--normalizer_fn', dest='normalizer_fn', default='None', choices=['batch_norm','group_norm', 'instance_norm', 'None'], help='normalization technique')

parser.add_argument('--resize', dest='resize', action='store_true', help='resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='scale images to this size')

parser.add_argument('--crop', dest='crop', action='store_true', help='crop input images, default no crop')
parser.set_defaults(crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='then crop to this size')

parser.add_argument('--not_full_summary', dest='full_summary', action='store_false', help='summary images')
parser.set_defaults(full_summary=True)

args = parser.parse_args()

params = Parameters(
    args.model,
    args.crop,
    args.crop_h, 
    args.crop_w, 
    args.resize, 
    args.resize_h, 
    args.resize_w, 
    args.data_path, 
    args.input_list_train, 
    args.task, 
    'train', 
    args.batch_size, 
    'False', 
    args.use_skips,
    args.normalizer_fn,
    args.num_classes,
    args.full_summary)

inputs = Dataloader(params).inputs
model = NetworkFactory.factory(inputs, params)

loss = model.loss
pred = model.pred

lr = tf.placeholder(tf.float32)

summary_image = tf.summary.merge(model.summary_images)
model.summary_scalar.append(tf.summary.scalar("lr", lr))
summary_scalar = tf.summary.merge(model.summary_scalar)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optim = tf.train.AdamOptimizer(lr, args.beta1).minimize(loss)

print('Finished building Network.')

writer = tf.summary.FileWriter(args.checkpoint_dir) 
saver = tf.train.Saver(max_to_keep=2)
saver_5000 = tf.train.Saver(max_to_keep=0)

init = [tf.global_variables_initializer(),tf.local_variables_initializer()]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)

    start_step = load(sess,args.checkpoint_dir)
    print("Loading last checkpoint")
    if  start_step >= 0:
        print("Restored step: ", start_step)
        print(" [*] Load SUCCESS")
    else:
        start_step=0
        print(" [!] Load failed...")   

    ### WRITING COMMAND LOG ###
    with open(os.path.join(args.checkpoint_dir, 'params.txt'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    print('Thread running')
    print('Running the Network')
    
    total_time = 0
    power = 0.9

    for step in range(start_step, args.steps + 1): 
        start_time = time.time()

        lr_value = args.lr*(1 - step/args.steps)**power
        loss_value, _ = sess.run([loss,optim],feed_dict={lr: lr_value})
        
        total_time += time.time() - start_time
        time_left = (args.steps - step - 1)*total_time/(step + 1 - start_step)

        if step%10==0:
            summary_string = sess.run(summary_scalar,feed_dict={lr: lr_value})
            writer.add_summary(summary_string,step)
            print("Step " , step, " loss: ",loss_value, "Time left: ", datetime.timedelta(seconds=time_left))

        if step%100==0 and args.full_summary and step!=0:
            summary_string = sess.run(summary_image)
            writer.add_summary(summary_string,step)
            print("Saved image summary",step)

        if step % 5000 ==0 and step!=0: 
            save(sess,saver_5000,os.path.join(args.checkpoint_dir, params.task + "_" + params.encoder),step=step)
            print("Saved checkpoint ", step)

        elif step % 1000 ==0 and step!=0:
            save(sess,saver,os.path.join(args.checkpoint_dir, params.task + "_" + params.encoder),step=step)
            print("Saved checkpoint ", step)
        
    coord.request_stop()
    coord.join()