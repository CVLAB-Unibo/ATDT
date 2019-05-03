import tensorflow as tf
from ops import *
from input import *
import argparse
from utils import *
import time
import datetime
import models
import sys
from NetworkFactory import TransferNetwork

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list', dest='input_list', default='input_list.txt', help='training or test relative path to each sample, gt separeted by ;')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

parser.add_argument('--steps', dest='steps', type=int, default=100000, help='# of steps')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')

parser.add_argument('--normalizer_fn', dest='normalizer_fn', default='None', choices=['batch_norm','group_norm', 'instance_norm', 'None'], help='normalization technique')
parser.add_argument("--model", default='dilated-resnet', choices=['vgg','resnet','dilated-resnet'], help='resnet, dilated-resnet, vgg')

parser.add_argument('--target_task', dest='target_task', choices=['semantic','depth', 'unsupervised-depth','normals'], help='[DECODER] target_task')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=19, help='[DECODER] # of classes')

parser.add_argument('--use_skips', dest='use_skips', action='store_true', help='use skip connection beetween encoder and decoder')
parser.set_defaults(use_skips=False)
parser.add_argument('--multiscale', dest='multiscale', action='store_true', help='multiscale')
parser.set_defaults(multiscale=False)

parser.add_argument('--feature_level', dest='feature_level', type=int, default=-1, help='feature level to align')

### ENCODER OPTION
parser.add_argument('--checkpoint_encoder_source', dest='checkpoint_encoder_source', default='', help='[ENCODER SOURCE] path to checkpoint folder or ckpt encoder')
parser.add_argument('--checkpoint_encoder_target', dest='checkpoint_encoder_target', default='', help='[ENCODER TARGET] path to checkpoint folder or ckpt encoder')
parser.add_argument('--checkpoint_decoder', dest='checkpoint_decoder', default='', help='[DECODER] path to checkpoint folder or ckpt transfer')

parser.add_argument('--resize', dest='resize', action='store_true', help='[ENCODER] resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='[ENCODER] scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='[ENCODER] scale images to this size')

parser.add_argument('--central_crop', dest='central_crop', action='store_true', help='[ENCODER] central_crop')
parser.set_defaults(central_crop=False)
parser.add_argument('--random_crop', dest='random_crop', action='store_true', help='[ENCODER] random_crop')
parser.set_defaults(random_crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='[ENCODER] then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='[ENCODER] then crop to this size')

args = parser.parse_args()

params = Parameters(
    args.encoder,
    args.central_crop or args.random_crop,
    args.crop_h, 
    args.crop_w, 
    args.resize, 
    args.resize_h, 
    args.resize_w, 
    args.data_path, 
    args.input_list, 
    args.target_task, 
    'train', 
    args.batch_size, 
    'False', 
    args.use_skips,
    args.normalizer_fn,
    args.num_classes,
    False)

inputs_all = Dataloader(params, False, central_crop=args.central_crop).inputs
inputs, gts = inputs_all

model = TransferNetwork(inputs, params)
loss = model.loss
pred = model.pred

var_list = [v for v in tf.trainable_variables() if "transfer" in v.name]
lr = tf.placeholder(tf.float32)
optim = tf.train.AdamOptimizer(lr,args.beta1).minimize(loss,var_list=var_list)

summary_image = tf.summary.merge(model.summary_images)
summary_scalar = tf.summary.merge(model.summary_scalar.append(tf.summary.scalar("lr", lr)))

writer = tf.summary.FileWriter(args.checkpoint_dir)

init = [tf.global_variables_initializer(),tf.local_variables_initializer()]

saver = tf.train.Saver(var_list=var_list,max_to_keep=2)
saver_5000 = tf.train.Saver(var_list=var_list,max_to_keep=0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)

    start_step = load(sess,args.checkpoint_dir)
    print("Loading last checkpoint")
    if  start_step >= 0:
        print("Restored Transfer step: ", start_step)
        print(" [*] Load SUCCESS")
    else:
        start_step=0
        print(" [!] Load failed...") 
    
    if args.checkpoint_encoder_source:
        success = load(sess,args.checkpoint_encoder_source,[], prefix="source/")
        if  success >= 0:
            print("Restored Encoder Source step: ", success)
        else:
            print("Failed Encoder Source Checkpoint Restore")
    else:
        print("No Encoder Checkpoint Source to Restore")

    if args.checkpoint_encoder_target:
        success = load(sess,args.checkpoint_encoder_target,[], prefix="target/")
        if  success >= 0:
            print("Restored Encoder Target step: ", success)
        else:
            print("Failed Encoder Target Checkpoint Restore")
    else:
        print("No Encoder Target Checkpoint to Restore")

    if args.checkpoint_decoder:
        mask=[]
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in variables:
            if "decoder" not in v.name[:-2]:
                mask.append(v.name[:-2])
        success = load(sess,args.checkpoint_decoder,mask)
        if  success >= 0:
            print("Restored Decoder step: ", success)
        else:
            print("Failed Decoder Checkpoint Restore")
    else:
        print("No Decoder Checkpoint to Restore")


    ### WRITING COMMAND LOG ###
    with open(os.path.join(args.checkpoint_dir, 'params.txt'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')
    
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners()
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

        if step%100==0 and args.full_summary:
            summary_string = sess.run(summary_image)
            writer.add_summary(summary_string,step)
            print("Saved image summary",step)

        if step % 5000 ==0: 
            save(sess,saver_5000,os.path.join(args.checkpoint_dir, "weights"),step=step)
            print("Saved checkpoint ", step)

        elif step % 1000 ==0:
            save(sess,saver,os.path.join(args.checkpoint_dir, "weights"),step=step)
            print("Saved checkpoint ", step)
        
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)
