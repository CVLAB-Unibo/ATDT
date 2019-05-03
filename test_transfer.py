import tensorflow as tf
from ops import *
from input import *
import argparse
from utils import *
import time
import datetime
import models
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list', dest='input_list', default='input_list.txt', help='training or test relative path to each sample, gt separeted by ;')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='outputs')

parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument("--feature_shape", help="[NO ENCODER] three ints encoding feature shape", nargs='+', type=int, default=[32,32,2048])

parser.add_argument('--normalizer_fn', dest='normalizer_fn', default='None', choices=['batch_norm','group_norm', 'instance_norm', 'None'], help='normalization technique')
parser.add_argument("--decoder", dest='decoder', default='None', choices=['vgg','resnet','dilated-resnet','None'], help='resnet, dilated-resnet, vgg or None')
parser.add_argument('--checkpoint_decoder', dest='checkpoint_decoder', default='', help='[DECODER] path to checkpoint folder or ckpt transfer')

parser.add_argument('--target_task', dest='target_task', choices=['semantic','depth', 'unsupervised-depth','normals'], help='[DECODER] target_task')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=19, help='[DECODER] # of classes')

parser.add_argument('--use_skips', dest='use_skips', action='store_true', help='use skip connection beetween encoder and decoder')
parser.set_defaults(use_skips=False)

parser.add_argument('--multiscale', dest='multiscale', action='store_true', help='multiscale')
parser.set_defaults(multiscale=False)

parser.add_argument('--feature_level', dest='feature_level', type=int, default=-1, help='feature level to align')

### ENCODER OPTION
parser.add_argument("--encoder", dest='encoder', default='None', choices=['vgg','resnet','dilated-resnet','None'], help='resnet, dilated-resnet, vgg or None')
parser.add_argument('--checkpoint_encoder_source', dest='checkpoint_encoder_source', default='', help='[ENCODER SOURCE] path to checkpoint folder or ckpt encoder')
parser.add_argument('--checkpoint_encoder_target', dest='checkpoint_encoder_target', default='', help='[ENCODER TARGET] path to checkpoint folder or ckpt encoder')

parser.add_argument('--resize', dest='resize', action='store_true', help='[ENCODER] resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='[ENCODER] scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='[ENCODER] scale images to this size')

parser.add_argument('--central_crop', dest='central_crop', action='store_true', help='[ENCODER] central_crop')
parser.set_defaults(central_crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='[ENCODER] then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='[ENCODER] then crop to this size')

parser.add_argument('--save_feature', dest='save_feature', action='store_true', help='save feature')
parser.set_defaults(save_feature=False)
parser.add_argument('--save_prediction', dest='save_prediction', action='store_true', help='save_prediction')
parser.set_defaults(save_prediction=False)

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
    args.target_task, 
    'test', 
    1, 
    'False', 
    args.use_skips,
    args.normalizer_fn,
    args.num_classes,
    False)

if args.normalizer_fn == 'batch_norm':
    normalizer_fn = lambda x : tf.layers.batch_normalization(x,training=False)
elif args.normalizer_fn == 'group_norm':
    normalizer_fn = lambda x : group_norm(x)
elif args.normalizer_fn == 'instance_norm':
    normalizer_fn = lambda x : instance_norm(x)
else: 
    normalizer_fn=None

def build_encoder(inputs, encoder, use_skips, force_level=0, feature_adapted=None):
    with tf.variable_scope('model'):
        if encoder == 'vgg':
            print("Building VGG Encoder")
            features, skips = models.build_vgg(inputs, use_skips, normalizer_fn=normalizer_fn, force_level=force_level, feature_adapted=feature_adapted)
        elif encoder == 'resnet':
            print("Building ResNet50 Encoder")
            features, skips = models.build_resnet50(inputs, use_skips, normalizer_fn=normalizer_fn, force_level=force_level, feature_adapted=feature_adapted)
        elif encoder == 'dilated-resnet':
            print("Building Dilated-Resnet Encoder")
            features, skips = models.build_dilated_resnet50(inputs, use_skips, normalizer_fn=normalizer_fn, force_level=force_level, feature_adapted=feature_adapted)
        return features, skips


def build_decoder(features, decoder, num_classes, target_task, normalizer_fn):
    if target_task == 'semantic':
        ch=num_classes
    elif target_task == 'depth':
        ch=1
    else:
        ch=3
    with tf.variable_scope('model'):
        if decoder == 'vgg':
            print("Building VGG Decoder")
            output = models.build_decoder_vgg(features, ch, normalizer_fn=normalizer_fn)
        elif decoder == 'resnet':
            print("Building ResNet50 Decoder")
            output = models.build_decoder_resnet(features, ch, normalizer_fn=normalizer_fn)
        elif decoder == 'dilated-resnet':
            print("Building Dilated-Resnet Decoder")
            output = models.build_decoder_dilated_resnet(features, ch, normalizer_fn=normalizer_fn)
    return output

summaries=[]

#### ENCODERS ####
if args.encoder != 'None':
    inputs_all = Dataloader(params, True).inputs
    inputs = inputs_all
    with tf.variable_scope('source'):
        features_source, skips = build_encoder(inputs,args.encoder,args.use_skips)
    with tf.variable_scope('target'):
        features_target, skips = build_encoder(inputs,args.encoder,args.use_skips)
        features_target= features_target[args.feature_level]
else:
    features_source, features_target = FeatureLoader(args.data_path,args.input_list,feature_shape=args.feature_shape,batch_size=args.batch_size,mode='test',shuffle=False).inputs

#### TRANSFER ####
with tf.variable_scope('transfer'):
    if args.multiscale and args.encoder != 'None':
        concats=[]
        for f in features_source[:-1]:
            # s = [features_source[-1].get_shape()[1],features_source[-1].get_shape()[2]]
            # print("Stride",s)
            # concats.append(tf.image.resize_images(f,s))
            s = f.get_shape()[1].value//features_source[-1].get_shape()[1].value
            print("Stride",s)
            concats.append(conv(f, f.get_shape()[-1], 3, s))
        concats.append(features_source[-1])
        features_source = tf.concat(concats,axis=-1)
    elif args.encoder != 'None':
        features_source = features_source[args.feature_level]

    print("Building Transfer Network")
    conv1 = conv(features_source,2048,3,2) ##16x16x2048
    conv2 = conv(conv1,2048,3,2) ##8x8x2048
    upconv1 = upconv(conv2,  2048, 3, 2) ##16x16x2048
    iconv1  = conv(upconv1,   2048, 3, 1) ##16x16x2048
    upconv2 = upconv(iconv1,  2048, 3, 2) ##32x32x2048
    adapted_feature  = conv(upconv2,   2048, 3, 1) ##32x32x2048

if args.encoder != 'None' and args.feature_level != -1:
    with tf.variable_scope('target', reuse=True):
        feat, skips = build_encoder(inputs,args.encoder,args.use_skips,force_level=args.feature_level, feature_adapted=adapted_feature )
        feat= feat[-1]
else:
    feat = adapted_feature

#### DECODERS ####
if args.decoder != 'None':
    output = build_decoder(feat, args.decoder, args.num_classes, args.target_task, normalizer_fn)
    if args.target_task=='normals':
        output=tf.nn.tanh(output)
else:
    output = feat

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
    
    if args.checkpoint_encoder_source and args.encoder != 'None':
        success = load(sess,args.checkpoint_encoder_source,[], prefix="source/")
        if  success >= 0:
            print("Restored Encoder Source step: ", success)
        else:
            print("Failed Encoder Source Checkpoint Restore")
    else:
        print("No Encoder Checkpoint Source to Restore")

    if args.checkpoint_encoder_target and args.encoder != 'None':
        success = load(sess,args.checkpoint_encoder_target,[], prefix="target/")
        if  success >= 0:
            print("Restored Encoder Target step: ", success)
        else:
            print("Failed Encoder Target Checkpoint Restore")
    else:
        print("No Encoder Target Checkpoint to Restore")

    if args.checkpoint_decoder and args.decoder != 'None':
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
        outputs_values = sess.run([adapted_feature, output])

        if args.encoder != 'None':
            basename, ext = os.path.splitext(lines[i].split(";")[0].replace("/","_"))
        else:
            basename, ext = os.path.splitext(lines[i].split(";")[0].split("/")[-1])

        if args.decoder != 'None' and args.save_prediction:
            dest_path = os.path.join(args.test_dir,"predictions", basename)
            if args.target_task == 'semantic':
                p=np.expand_dims(np.argmax(outputs_values[1],axis=-1)[0],axis=-1).astype(np.uint8)
                cv2.imwrite(dest_path + ".png" , p)
            elif args.target_task == 'normals':
                p=((outputs_values[1][0]+1)/2*255).astype(np.uint8)
                cv2.imwrite(dest_path + ".png" , cv2.cvtColor(p,cv2.COLOR_RGB2BGR))
            else:
                np.save(dest_path + ".npy", outputs_values[1])

        if args.save_feature:
            dest_path = os.path.join(args.test_dir, "features", basename + ".npz")
            np.savez_compressed(dest_path, outputs_values[0][0])
        
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)