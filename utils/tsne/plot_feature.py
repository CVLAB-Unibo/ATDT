import os
import sys
import numpy as np
import tensorflow as tf
import datetime
import argparse
import cv2
import time
from tensorflow.contrib.tensorboard.plugins import projector

from utils import *
import NetworkFactory
from input import Dataloader
import ops
import math

SPRITE_FILENAME_RGB = 'sprite_rgb.png'
SPRITE_FILENAME_PRED = 'sprite_pred.png'
METADATA_FILENAME = 'metadata.tsv'
NAMEMAP_FILENAME = 'namemap.tsv'
EMBEDDING_FILENAME = 'embedding.ckpt'
EMBEDDING_NPY = 'embedding.npy'
MAX_SPRITE_SIDE = 8192
SPRITE_SHAPE = [32,32,3]

def ratio_preserving_resize(immy,target=256):
    ss = immy.shape
    if ss[0]>ss[1]:
        new_dim=target*ss[1]//ss[0]
        resize_dim = (new_dim,target)
    else:
        new_dim=target*ss[0]//ss[1]
        resize_dim = (target,new_dim)
    new_immy=cv2.resize(immy,resize_dim)

    pad_size=target-new_dim
    pad_left=pad_size//2
    pad_right=pad_size//2+pad_size%2

    if ss[0]>ss[1]:
        pad=((0,0),(pad_left,pad_right),(0,0))
    else:
        pad=((pad_left,pad_right),(0,0),(0,0))
    new_immy=np.pad(new_immy,pad,mode='constant',constant_values=125)
    new_immy=new_immy[:,:,::-1]

    return new_immy,pad[:2]

def create_metadata(input_list, destination_file):
    """
    Write filename and dataset in a metadata file
    CityScape == 2
    Synthia == 1 
    Syncity == 0
    """
    def decode_dataset(line):
        if "City" in line:
            #syncity
            return 0
        elif "SEQ" in line:
            #synthia
            return 1
        elif "_leftImg8bit.png" in line:
            #cityscape
            return 2
        elif "episode_" in line:
            return 3
     
    out_lines = []
    out_lines.append('Name\tDataset\n')
    with open(input_list) as f_in:
        lines = f_in.readlines()
        for l in lines:
            dataset = decode_dataset(l)
            filename = l.split(';')[0].replace('/','_')
            out_lines.append('{}\t{}\n'.format(filename,dataset))
    
    with open(destination_file,'w+') as f_out:
        f_out.writelines(out_lines)

def create_sprites(image_list,sprite_shapes,output_file):
    num_image = len(image_list)
    image_per_side = math.ceil(math.sqrt(num_image))

    sprite = np.zeros([sprite_shapes[0]*(image_per_side),sprite_shapes[1]*image_per_side,3])
    for c,im in enumerate(image_list):
        try:
            im,_ = ratio_preserving_resize(im,sprite_shapes[0])
        except IOError:
            print("cannot create thumbnail for '%s'"%infile)
        start_row = (c//image_per_side) * sprite_shapes[0]
        start_col = (c%image_per_side) * sprite_shapes[1]
        sprite[start_row:start_row+sprite_shapes[0],start_col:start_col+sprite_shapes[1],:] = im[:,:,0:3]
    
    #save array to output_file
    cv2.imwrite(output_file,sprite)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list',  default='input_list_val_test.txt', help='path of the input pair for validation or testing, image\\timage_sem')
parser.add_argument('--network_weights', default='./checkpoint', help='checkpoint folder or path for the task network')
parser.add_argument('--transfer_weights', help="checkpoint for the transfer network")
parser.add_argument('--out_dir', default='./test/', help='features goes here')

parser.add_argument('--num_classes', dest='num_classes', type=int, default=19, help='# of classes')
parser.add_argument('--encoder', dest='encoder', default='dilated-resnet', choices=['vgg','resnet','dilated-resnet'], help='resnet, dilated-resnet or vgg')
parser.add_argument('--task', dest='task', required=True, choices=['semantic','depth', 'unsupervised-depth'], help='task')

parser.add_argument('--transfer', help="flag to enable transfer function", action="store_true")
parser.add_argument('--use_skips', dest='use_skips', action='store_true', help='use skip connection beetween encoder and decoder')
parser.set_defaults(use_skips=False)
parser.add_argument('--normalizer_fn', dest='normalizer_fn', default='None', choices=['batch_norm','group_norm', 'instance_norm', 'None'], help='normalization technique')

parser.add_argument('--resize', dest='resize', action='store_true', help='resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='scale images to this size')

parser.add_argument('--central_crop', dest='central_crop', action='store_true', help='central_crop')
parser.set_defaults(central_crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='then crop to this size')

parser.add_argument('--feature_level',help="depth of the feature we want to plot", default=-1, type=int)

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
    args.input_list, 
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
model.build()

pred = model.pred
features = model.features[args.feature_level]
images = inputs[0]

if args.transfer:
    #### TRANSFER ####
    with tf.variable_scope('transfer'):
        print("Building Transfer Network")
        conv1 = ops.conv(features,2048,3,2) ##16x16x2048
        conv2 = ops.conv(conv1,2048,3,2) ##8x8x2048
        upconv1 = ops.upconv(conv2,  2048, 3, 2) ##16x16x2048
        iconv1  = ops.conv(upconv1,   2048, 3, 1) ##16x16x2048
        upconv2 = ops.upconv(iconv1,  2048, 3, 2) ##32x32x2048
        features  = ops.conv(upconv2,   2048, 3, 1) ##32x32x2048

init = [tf.global_variables_initializer(),tf.local_variables_initializer()]

saver = tf.train.Saver()

with tf.Session() as sess:
    ### CREATE OUTPUT FOLDER
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    #init    
    sess.run(init)

    step_net = load(sess,args.network_weights, prefix="")
    print("Loading last checkpoint")
    if  step_net >= 0:
        print("Restored step: ", step_net)
        print(" [*] Load SUCCESS")
    else:
        step_net=0
        print(" [!] Load failed...")  
        raise Exception('Load Failed') 

    if args.transfer:
        step_tr = load(sess, args.transfer_weights)
        print("Loading last checkpoint")
        if  step_tr >= 0:
            print("Restored step: ", step_tr)
            print(" [*] Load SUCCESS")
        else:
            step_tr = 0
            print(" [!] Load failed...")  
            raise Exception('Load Failed') 

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners()
    print('Thread running')
    print('Running the Network')

    lines = open(args.input_list).readlines()
    num_sample = len(lines)

    #extract embedding vectors
    for i in range(num_sample):
        print(i,"/",num_sample,end='\r')
        temp_feature, prediction, rgb_input = sess.run([features,pred,images])
        temp_feature = np.squeeze(np.mean(temp_feature,axis=(0,1,2))).ravel() 
        if i == 0:
            rgb_images = []
            predictions = []
            embedding = np.ndarray(shape=(num_sample,temp_feature.shape[0]), dtype=float)
        embedding[i] = temp_feature
        rgb_images.append(rgb_input*255)
        predictions.append(prediction[0])
    
    coord.request_stop()
    coord.join()

    #save backup copy as npy
    np.save(os.path.join(args.out_dir,EMBEDDING_NPY),embedding)
    print("Saved Backup")
    #create embedding var inside tf and initialize it
    
    with tf.variable_scope('embedding'):
        encoding = tf.Variable(embedding, trainable=False)
    sess.run(encoding.initializer)
    print("Initialized variable")
    #create metadata file
    create_metadata(args.input_list,os.path.join(args.out_dir,METADATA_FILENAME))
    print("Saved Metadata")
    #create sprite for rgb
    create_sprites(rgb_images,SPRITE_SHAPE, os.path.join(args.out_dir,SPRITE_FILENAME_RGB))
    print("Created Sprite RGB")
    #create sprite for semantic
    create_sprites(predictions,SPRITE_SHAPE, os.path.join(args.out_dir,SPRITE_FILENAME_PRED))
    print("Created Sprite Predictions")

    #Add Metadata
    summary_writer = tf.summary.FileWriter(args.out_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = encoding.name
    embedding.metadata_path = METADATA_FILENAME
    embedding.sprite.image_path = SPRITE_FILENAME_RGB
    embedding.sprite.single_image_dim.extend(SPRITE_SHAPE[0:2])
    projector.visualize_embeddings(summary_writer, config)

    #save embedding
    emb_saver = tf.train.Saver(var_list=[encoding])
    emb_saver.save(sess, os.path.join(args.out_dir, EMBEDDING_FILENAME), 0)
    print('Embedding saved')