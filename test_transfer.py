import tensorflow as tf
import argparse
import time
import datetime
import sys

from core.ops import *
from core.input import *
from utils.utils import *
import core.models as models
from core.NetworkFactory import TransferNetwork

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list', dest='input_list', default='input_list.txt', help='training or test relative path to each sample, gt separeted by ;')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='outputs')

parser.add_argument('--normalizer_fn', dest='normalizer_fn', default='None', choices=['batch_norm','group_norm', 'instance_norm', 'None'], help='normalization technique')
parser.add_argument("--model", dest='model', default='dilated-resnet', choices=['vgg','resnet','dilated-resnet','None'], help='resnet, dilated-resnet, vgg or None')

parser.add_argument('--target_task', dest='target_task', choices=['semantic','depth','normals'], help='[DECODER] target_task')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=11, help='[DECODER] # of classes')

parser.add_argument('--use_skips', dest='use_skips', action='store_true', help='use skip connection beetween encoder and decoder')
parser.set_defaults(use_skips=False)
parser.add_argument('--multiscale', dest='multiscale', action='store_true', help='multiscale')
parser.set_defaults(multiscale=False)
parser.add_argument('--feature_level', dest='feature_level', type=int, default=-1, help='feature level to align')

### ENCODER OPTION
parser.add_argument('--checkpoint_encoder_source', dest='checkpoint_encoder_source', default='', help='[ENCODER SOURCE] path to checkpoint folder or ckpt encoder')
#parser.add_argument('--checkpoint_encoder_target', dest='checkpoint_encoder_target', default='', help='[ENCODER TARGET] path to checkpoint folder or ckpt encoder')
parser.add_argument('--checkpoint_decoder', dest='checkpoint_decoder', default='', help='[DECODER] path to checkpoint folder or ckpt transfer')

parser.add_argument('--resize', dest='resize', action='store_true', help='[ENCODER] resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='[ENCODER] scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='[ENCODER] scale images to this size')

parser.add_argument('--central_crop', dest='central_crop', action='store_true', help='[ENCODER] central_crop')
parser.set_defaults(central_crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='[ENCODER] then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='[ENCODER] then crop to this size')

parser.add_argument('--save_features', dest='save_feature', action='store_true', help='save feature')
parser.set_defaults(save_feature=False)
parser.add_argument('--save_predictions', dest='save_prediction', action='store_true', help='save_prediction')
parser.set_defaults(save_prediction=False)

args = parser.parse_args()

params = Parameters(
    args.model,
    args.central_crop,
    args.crop_h, 
    args.crop_w, 
    args.resize, 
    args.resize_h, 
    args.resize_w, 
    args.data_path, 
    args.input_list, 
    args.target_task, 
    'test', 
    1, 
    'False', 
    args.use_skips,
    args.normalizer_fn,
    args.num_classes,
    False)

inputs = Dataloader(params, True).inputs
model = TransferNetwork(inputs, params, encoder_target=False)
adapted_features = model.adapted_features
pred = model.pred

init = [tf.global_variables_initializer(),tf.local_variables_initializer()]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)

    start_step = load(sess,args.checkpoint_dir)#, prefix="transfer/")
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

    # if args.checkpoint_encoder_target and args.encoder != 'None':
    #     success = load(sess,args.checkpoint_encoder_target,[], prefix="target/")
    #     if  success >= 0:
    #         print("Restored Encoder Target step: ", success)
    #     else:
    #         print("Failed Encoder Target Checkpoint Restore")
    # else:
    #     print("No Encoder Target Checkpoint to Restore")

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

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners()
    print('Thread running')
    print('Running the Network')
    
    lines = open(args.input_list).readlines()
    num_sample = len(lines)
    
    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)
    if not os.path.exists(os.path.join(args.test_dir,"features")):
        os.mkdir(os.path.join(args.test_dir,"features"))
    if not os.path.exists(os.path.join(args.test_dir,"predictions")):
        os.mkdir(os.path.join(args.test_dir,"predictions"))  


    ### WRITING COMMAND LOG ###
    with open(os.path.join(args.test_dir, 'params.txt'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')

    for i in range(num_sample):
        print(i,"/",num_sample,end='\r')

        start_time = time.time()
        outputs_values = sess.run([adapted_features, pred])

        basename, ext = os.path.splitext(lines[i].split(";")[0].replace("/","_"))

        if args.save_prediction:
            dest_path = os.path.join(args.test_dir,"predictions", basename)
            if args.target_task == 'semantic':
                #p=np.expand_dims(np.argmax(outputs_values[1],axis=-1)[0],axis=-1).astype(np.uint8)
                cv2.imwrite(dest_path + ".png" , outputs_values[1][0])
            elif args.target_task == 'normals':
                #p=((outputs_values[1][0]+1)/2*255).astype(np.uint8)
                cv2.imwrite(dest_path + ".png" , cv2.cvtColor(outputs_values[1][0],cv2.COLOR_RGB2BGR))
            else:
                np.save(dest_path + ".npy", outputs_values[1])

        if args.save_feature:
            dest_path = os.path.join(args.test_dir, "features", basename + ".npz")
            np.savez_compressed(dest_path, outputs_values[0][0])
        
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)