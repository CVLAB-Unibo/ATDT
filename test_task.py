import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import cv2
import time

from utils.utils import *
import core.NetworkFactory as NetworkFactory
from core.input import Dataloader

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list_val_test', dest='input_list_val_test', default='input_list_val_test.txt', help='path of the input pair for validation or testing, image\\timage_sem')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', default='./checkpoint', help='checkpoint folder or path')
parser.add_argument('--test_dir', dest='test_dir', default='./test/', help='test sample are saved here')

parser.add_argument('--num_classes', dest='num_classes', type=int, default=11, help='# of classes')
parser.add_argument('--encoder', dest='encoder', default='dilated-resnet', choices=['vgg','resnet','dilated-resnet'], help='resnet, dilated-resnet or vgg')
parser.add_argument('--task', dest='task', required=True, choices=['semantic','depth', 'unsupervised-depth','normals'], help='task')
parser.add_argument('--use_skips', dest='use_skips', action='store_true', help='use skip connection beetween encoder and decoder')
parser.set_defaults(use_skips=False)
parser.add_argument('--normalizer_fn', dest='normalizer_fn', default='None', choices=['batch_norm','group_norm', 'instance_norm', 'None'], help='normalization technique')

parser.add_argument('--save_features', dest='save_features', action='store_true', help='save activations as npy')
parser.set_defaults(save_features=False)
parser.add_argument('--save_predictions', dest='save_predictions', action='store_true', help='save predictions')
parser.set_defaults(save_predictions=False)
parser.add_argument('--save_semantic_volume', dest='save_semantic_volume', action='store_true', help='save_semantic_volume')
parser.set_defaults(save_semantic_volume=False)

parser.add_argument('--resize', dest='resize', action='store_true', help='resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='scale images to this size')

parser.add_argument('--central_crop', dest='central_crop', action='store_true', help='central_crop')
parser.set_defaults(central_crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='then crop to this size')

args = parser.parse_args()

params = Parameters(
    args.encoder,
    args.central_crop,
    args.crop_h, 
    args.crop_w, 
    args.resize, 
    args.resize_h, 
    args.resize_w, 
    args.data_path, 
    args.input_list_val_test, 
    args.task, 
    'test', 
    1, 
    'False', 
    args.use_skips,
    args.normalizer_fn,
    args.num_classes,
    False)


inputs = Dataloader(params,True).image_batch

model = NetworkFactory.factory(inputs, params)
pred = model.pred_map
features = model.features[-1]
images = inputs[0]

init = [tf.global_variables_initializer(),tf.local_variables_initializer()]

saver = tf.train.Saver()

with tf.Session() as sess:
    ### CREATE OUTPUT FOLDER
    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)
    if args.save_features and not os.path.exists(os.path.join(args.test_dir,"features")):
        os.mkdir(os.path.join(args.test_dir,"features"))
    if args.save_predictions and not os.path.exists(os.path.join(args.test_dir,"predictions")):
        os.mkdir(os.path.join(args.test_dir,"predictions"))        
    if args.save_semantic_volume and not os.path.exists(os.path.join(args.test_dir,"semantic_volume")):
        os.mkdir(os.path.join(args.test_dir,"semantic_volume"))

    sess.run(init)

    start_step = load(sess,args.checkpoint_path)
    print("Loading last checkpoint")
    if  start_step >= 0:
        print("Restored step: ", start_step)
        print(" [*] Load SUCCESS")
    else:
        start_step=0
        print(" [!] Load failed...")   

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners()
    print('Thread running')
    print('Running the Network')

    ### WRITING COMMAND LOG ###
    with open(os.path.join(args.test_dir, 'params.txt'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')
    
    lines = open(args.input_list_val_test).readlines()
    num_sample = len(lines)

    for i in range(num_sample):
        print(i,"/",num_sample,end='\r')
        outputs = []
        if args.save_features:
            outputs.append(features)
        if args.save_predictions:
            outputs.append(pred)
        if args.save_semantic_volume:
            outputs.append(model.logits)
        
        start_time = time.time()
        outputs_values = sess.run(outputs)
        
        tot=0
        if args.save_features:
            dest_path = os.path.join(args.test_dir, "features", lines[i].split(";")[0].replace("/","_").replace(".png",".npz"))
            np.savez_compressed(dest_path, outputs_values[tot])
            tot += 1
        if args.save_predictions:
            dest_path = os.path.join(args.test_dir, "predictions", lines[i].split(";")[0].replace("/","_"))
            prediction = outputs_values[tot]
            if args.task == 'semantic':
                prediction = prediction.astype(np.uint8)
                cv2.imwrite(dest_path,prediction[0])
            elif args.task == 'normals':
                p=((prediction[0]+1)/2*255).astype(np.uint8)
                cv2.imwrite(dest_path + ".png" , cv2.cvtColor(p,cv2.COLOR_RGB2BGR))
            elif args.task == 'depth':
                dest_path = dest_path.replace(".png",".npy") 
                np.save(dest_path, prediction)
            tot += 1
        if args.save_semantic_volume:
            dest_path = os.path.join(args.test_dir, "semantic_volume", lines[i].split(";")[0].replace("/","_").replace(".png",".npz"))
            np.savez_compressed(dest_path, outputs_values[tot])

    print("Completed.")
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)