from collections import namedtuple
import cv2
import sys
import numpy as np
import os
import matplotlib
import argparse

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels_cityscapes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

labels_default = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  1 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             ,  2 ,        9 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 ,  3 ,        2 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                ,  4 ,        3 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 ,  5 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        ,  6 ,        8 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         ,  7 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           ,  8 ,        6 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              ,  9 ,      255 , 'nature'          , 4       , False        , False        , (0,0,0) ),
    Label(  'sky'                  , 10 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 11 ,        4 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 12 ,      255 , 'human'           , 6       , True         , False        , (0,  0,  0) ),
    Label(  'car'                  , 13 ,        7 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 14 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0, 0) ),
    Label(  'bus'                  , 15 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 0,0) ),
    Label(  'train'                , 16 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 0,0) ),
    Label(  'motorcycle'           , 17 ,        7 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 18 ,      255 , 'vehicle'         , 7       , True         , False        , (0, 0, 0) ),
]

parser = argparse.ArgumentParser(description = 'convert labels')
parser.add_argument('--input_path', type=str,   help='path to the folder or to the single file to convert, or path to the txt file with all absolute paths to images ', required=True)
parser.add_argument('--output_path', type=str,   help='path to output folder ', default="./output_converted")
parser.add_argument('--from_encoding', type=str,   help='from which encoding', choices=['id','trainId'], default="id")
parser.add_argument('--to_encoding', type=str,   help='to which encoding [id,trainId,color], multiple choice allowed divided by ,', default="color")
parser.add_argument('--dataset', type=str, default="default")
args=parser.parse_args()

if args.dataset=='default':
    labels=labels_default
else:
    labels=labels_cityscapes

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

to = args.to_encoding.strip().split(",")
for e in to:
    if e not in ['id','trainId','color']:
        print(e, "wrong encoding, ignored")

d = {}
d_color = {}

if args.from_encoding == "id":
    if 'trainId' in to:
        d = { label.id  : label.trainId for label in labels }
    if "color" in to:
        d_color = { label.id    : label.color for label in labels }

if args.from_encoding == 'trainId':
    if 'id' in to:
        d = {label.trainId : label.id for label in labels if not label.ignoreInEval}
    if 'color' in to:
        d_color = {label.trainId : label.color for label in labels if not label.ignoreInEval}

lut=None
lutColor=None
if 'color' in to:
    zeros = np.zeros(256, np.dtype('uint8'))
    lutColor = np.dstack((zeros, zeros, zeros))
    for i in range(0,255):
        if i in d_color.keys():
            lutColor[0][i] = [d_color[i][2],d_color[i][1],d_color[i][0]]
    lutColor = np.array(lutColor)
    
if 'id' in to or 'trainId' in to:
    lut = [255]*256
    for i in range(0,255):
        if i in d.keys():
            lut[i] = d[i]
    lut=np.array(lut)

cnt = 0
lenght =len(os.listdir(args.input_path))
for file in os.listdir(args.input_path):
    if ".png" in file:
        cnt = cnt+1
        print("File: " , cnt , "/" , lenght, end='\r')
        file_path = os.path.join(args.input_path,file)
        file_output_path = os.path.join(args.output_path,file)
        img = cv2.imread(file_path)
        if lut is not None:
            img_sem = cv2.LUT(img,lut).astype(np.uint8)
            cv2.imwrite(file_output_path,cv2.cvtColor(img_sem,cv2.COLOR_BGR2GRAY))
        if lutColor is not None:
            img_sem_color = cv2.LUT(img,lutColor)
            file_out = os.path.splitext(file_output_path)
            cv2.imwrite(file_out[0] + "_color" + file_out[1] ,img_sem_color)

print("\nElaborated ",lenght," files")

