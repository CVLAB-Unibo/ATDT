import numpy as np
import argparse
import os
import cv2
import tensorflow as tf
from collections import namedtuple
import math

from utils.utils import *

parser = argparse.ArgumentParser(description='Evaluation on the cityscapes validation set')
parser.add_argument('--dataset_source', dest='dataset_source', choices=['synthia','syncity','kitti','cityscapes','carla'], default=None, help='synthia,syncity')
parser.add_argument('--dataset_target', dest='dataset_target', choices=['synthia','syncity','kitti','cityscapes','carla'], help='synthia,syncity')

parser.add_argument('--data_path', dest='data_path',  help='absolute path to dataset containing folder')
parser.add_argument('--input_list_val_test', dest='input_list_val_test', default='input_list_val_test.txt', help='path of the input pair for validation or testing, image\\timage_sem')
parser.add_argument('--task', dest='task', required=True, choices=['semantic','depth', 'unsupervised-depth'], help='task')
parser.add_argument('--pred_folder',  type=str,   help='folder containing predictions semantic maps',   required=True)
parser.add_argument('--output_path',  type=str, default='results.txt', help='output results.txt')

parser.add_argument('--resize', dest='resize', action='store_true', help='resize input images and gt to same size, default no resize')
parser.set_defaults(resize=False)
parser.add_argument('--resize_w', dest='resize_w', type=int, default=-1, help='scale images to this size')
parser.add_argument('--resize_h', dest='resize_h', type=int, default=-1, help='scale images to this size')

parser.add_argument('--central_crop', dest='central_crop', action='store_true', help='central_crop')
parser.set_defaults(central_crop=False)
parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='then crop to this size')
parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='then crop to this size')

### SEMANTIC PARAMS
parser.add_argument('--num_classes', dest='num_classes', type=int, default=11, help='[SEMANTIC] # of classes')
parser.add_argument('--format_pred', type=str, choices=['id','trainId'], default='trainId',help='[SEMANTIC] encoding of predictions, trainId or id')
parser.add_argument('--format_gt', type=str, choices=['id','trainId'], default='trainId',help='[SEMANTIC] encoding of gt, trainId or id')
parser.add_argument('--ignore_label', type=int, default=255, help='[SEMANTIC] label to ignore in evaluation')
parser.add_argument('--convert_to', dest='convert_to', choices=['synthia','syncity','kitti','cityscapes','carla'], help='[SEMANTIC] synthia,syncity, kitti,cityscapes,carla')
parser.add_argument('--convert_pred', dest='convert_pred', action='store_true',help="[SEMANTIC] convert prediction to target classes")
parser.add_argument('--convert_gt', dest='convert_gt', action='store_true', help="[SEMANTIC] convert ground truth to target classes")
parser.set_defaults(convert_pred=False)

### DEPTH PARAMS
parser.add_argument('--min_depth',           type=float, help='[DEPTH] minimum depth for evaluation in m',        default=0.001)
parser.add_argument('--max_depth',           type=float, help='[DEPTH] maximum depth for evaluation in m',        default=100)
# parser.add_argument('--f_source',           type=float, help='[DEPTH] focal lenght A',        default=859.238022326)
# parser.add_argument('--f_target',           type=float, help='[DEPTH] focal lenght B',        default=859.238022326)

args = parser.parse_args()

dict_focals={
    'cityscapes': 2262.52,
    'carla': 859.238022326,
    'synthia': 847.630211643,
    'kitti': -1}

if args.dataset_source == None:
    focal_source=dict_focals[args.dataset_target]
else:
    focal_source=dict_focals[args.dataset_source]

focal_target=dict_focals[args.dataset_target]

id2trainId = { label.id : label.trainId for label in labels }
id2name =  { label.id : label.name for label in labels }

if args.dataset_target=='carla' or args.convert_to=='carla':
    trainId2name = { label.trainId : label.name for label in labels_cityscapes_to_carla }
else:
    trainId2name = { label.trainId : label.name for label in labels }

def convert_to_train_id(sem,id2trainId=id2trainId):
    p = tf.cast(sem,tf.uint8)
    m = tf.zeros_like(p)
    for i in range(0, len(labels)):
        mi = tf.multiply(tf.ones_like(p), id2trainId[i])
        m = tf.where(tf.equal(p,i), mi, m)
    return m

def convert_to_carla(toconvert):
    masks=[]
    for i in range(len(labels)):
        masks.append(np.where(np.equal(toconvert,np.ones_like(toconvert)*i), np.ones_like(toconvert)*cityscapes2carla[i], 0))
    masks.append(np.where(np.equal(toconvert,np.ones_like(toconvert)*255), 255, 0))
    masks=np.asarray(masks)
    toconvert = np.sum(masks,axis=0)
    return toconvert

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

### INPUTS ###
if args.task == 'depth' or args.task == 'unsupervised-depth':
    prediction_placeholder = tf.placeholder(tf.float32,shape=[None,None,1])
    gt_placeholder = tf.placeholder(tf.float32,shape=[None,None,1])
    resize_method = tf.image.ResizeMethod.BILINEAR
elif args.task == 'semantic':
    prediction_placeholder = tf.placeholder(tf.int32, shape=[None,None,1])
    gt_placeholder = tf.placeholder(tf.int32,shape=[None,None,1])
    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

gt = gt_placeholder
prediction = prediction_placeholder

### RESIZE ###
if args.resize:
    prediction = tf.image.resize_images(prediction, [args.resize_h, args.resize_w] ,method=resize_method)
    gt = tf.image.resize_images(gt, [args.resize_h, args.resize_w] ,method=resize_method)
### CROP ###
if args.central_crop:
    prediction = tf.image.resize_image_with_crop_or_pad(prediction,args.crop_h,args.crop_w)
    gt = tf.image.resize_image_with_crop_or_pad(gt,args.crop_h,args.crop_w)

if args.task == 'semantic':
    ### CONVERT TO IGNORE LABELS IN EVAL ###
    if args.format_pred == 'id':
        prediction = convert_to_train_id(prediction)
    if args.format_gt == 'id':
        gt = convert_to_train_id(gt)
    ### INIT WEIGHTS MIOU
    weightsValue = tf.to_float(tf.not_equal(gt,args.ignore_label))
    ### IGNORE LABELS TO 0, WE HAVE ALREADY MASKED THOSE PIXELS WITH WEIGHTS 0###
    gt = tf.where(tf.equal(gt, args.ignore_label), tf.zeros_like(gt), gt)
    prediction = tf.where(tf.equal(prediction, args.ignore_label), tf.zeros_like(prediction), prediction)
    ### ACCURACY ###
    acc, update_op_acc = tf.metrics.accuracy(gt,prediction,weights=weightsValue)
    ### MIOU ###
    miou, update_op = tf.metrics.mean_iou(labels=tf.reshape(gt,[-1]),predictions=tf.reshape(prediction,[-1]), num_classes=args.num_classes, weights=tf.reshape(weightsValue,[-1]))

 ### INIT OP ###
init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

miou_value = 0
with tf.Session() as sess:
    sess.run(init_op)
    with open(args.input_list_val_test) as filelist:
        lines = filelist.readlines()
        lenght = len(lines)
        
        if args.task == 'depth' or args.task == 'unsupervised-depth':
            rms     = np.zeros(lenght, np.float32)
            log_rms = np.zeros(lenght, np.float32)
            abs_rel = np.zeros(lenght, np.float32)
            sq_rel  = np.zeros(lenght, np.float32)
            a1      = np.zeros(lenght, np.float32)
            a2      = np.zeros(lenght, np.float32)
            a3      = np.zeros(lenght, np.float32)
            f_ratio=focal_target/focal_source

        for idx,line in enumerate(lines):
            print(idx, "/", lenght, end='\r')
            img_path = line.split(";")[0].strip()
            pred_path = os.path.join(args.pred_folder, img_path.replace("/","_"))
            
            if args.task == 'depth' or args.task == 'unsupervised-depth':
                if args.dataset_target=='kitti':
                    id_img=img_path.split("/")[-1].split("_")[0]
                    f_target=float(open(os.path.join(args.data_path,"calib",id_img + ".txt")).readlines()[0].split(" ")[1])
                    f_ratio=f_target/(fpcal_source)

                pred_path = pred_path.replace(".png",".npy")
                gt_path = os.path.join(args.data_path, line.split(";")[3].strip())

                pred_value = np.load(pred_path)

                if len(pred_value.shape) == 4:
                    pred_value = np.squeeze(pred_value,axis=0)
                
                pred_value = pred_value * args.max_depth * f_ratio
                pred_value[pred_value < args.min_depth] = args.min_depth
                pred_value[pred_value > args.max_depth] = args.max_depth

                ### READING GT BY DATASET
                if args.dataset_target == 'syncity':
                    f=100 #depth in m
                    gt_value = np.expand_dims(cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)[:,:,-1]*f,axis=-1)
                else:
                    raw = cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
                    raw = raw[:,:,:3].astype(np.float32)
                    f= 1000 #depth in m
                    gt_value = raw[:,:,0]*256*256 + raw[:,:,1]*256 + raw[:,:,2]
                    gt_value = gt_value / (256*256*256 - 1)
                    gt_value = np.expand_dims(gt_value*f,axis=-1)

                pred_value, gt_value = sess.run([prediction,gt],feed_dict={prediction_placeholder :pred_value , gt_placeholder : gt_value})
                mask = np.logical_and(gt_value > args.min_depth, gt_value < args.max_depth)
                abs_rel[idx], sq_rel[idx], rms[idx], log_rms[idx], a1[idx], a2[idx], a3[idx] = compute_errors(gt_value[mask], pred_value[mask])

            elif args.task == 'semantic':
                gt_path = os.path.join(args.data_path,line.split(";")[2].strip())
                pred_value = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)
                gt_value = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
                if args.convert_to=='carla':
                    if args.convert_gt:
                        gt_value = convert_to_carla(gt_value)
                    if args.convert_pred:
                        pred_value = convert_to_carla(pred_value)

                _,_ =sess.run([update_op_acc,update_op],feed_dict={prediction_placeholder : np.expand_dims(pred_value,axis=-1) , gt_placeholder : np.expand_dims(gt_value,axis=-1)})
                acc_value, miou_value =sess.run([acc, miou],feed_dict={prediction_placeholder : np.expand_dims(pred_value,axis=-1) , gt_placeholder : np.expand_dims(gt_value,axis=-1)})
        
        ### OUTPUT RESULTS
        output_file = open(args.output_path,"w")
        if args.task == 'depth' or args.task == 'unsupervised-depth':
            output_file.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3') + "\n")
            output_file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()) + "\n")
        elif args.task == 'semantic':
            confusion_matrix=tf.get_default_graph().get_tensor_by_name("mean_iou/total_confusion_matrix:0").eval()
            for cl in range(confusion_matrix.shape[0]):
                tp_fn = np.sum(confusion_matrix[cl,:])
                tp_fp = np.sum(confusion_matrix[:,cl])
                tp = confusion_matrix[cl,cl]
                if tp == 0 and (tp_fn + tp_fp - tp) == 0:
                    IoU_cl = float('nan')
                else:
                    IoU_cl = tp / (tp_fn + tp_fp - tp)
                output_file.write(trainId2name[cl] + ": {:.8f}".format(IoU_cl)+"\n")
            output_file.write("mIoU: " + str(miou_value) + " acc " + str(acc_value)+"\n")
