import numpy as np
import tensorflow as tf
import os
from collections import namedtuple
import cv2
import numpy 
import matplotlib.pyplot as plt

Parameters = namedtuple('Parameters',[
                        'encoder',
                        'crop',
                        'crop_h',
                        'crop_w',
                        'resize',
                        'resize_h',
                        'resize_w',
                        'data_path',
                        'filenames_file',
                        'task',
                        'mode',
                        'batch_size',
                        'use_deconv',
                        'use_skips',
                        'normalizer_fn',
                        'num_classes',
                        'full_summary',
                        ])

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

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
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
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]

labels_cityscapes_to_carla = [
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
    Label(  'terrain'              ,  9 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 10 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 11 ,        4 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 12 ,      255 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 13 ,        7 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 14 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 15 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 16 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 17 ,        7 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 18 ,      255 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]

trainId2Color = { label.trainId : label.color for label in labels }
cityscapes2carla= { label.id : label.trainId for label in labels_cityscapes_to_carla }

def color_tensorflow(pred_sem, id2color=trainId2Color):
    p = tf.squeeze(tf.cast(pred_sem,tf.uint8), axis = -1)
    p = tf.stack([p,p,p],axis=-1)
    m = tf.zeros_like(p)
    for i in range(len(trainId2Color.keys()) - 1):
        mi = tf.multiply(tf.ones_like(p), trainId2Color[i])
        m = tf.where(tf.equal(p,i), mi, m)
    return m

def colormap_depth(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size,height, width,1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    
    Returns a 3D tensor of shape [batch_size,height, width,3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = tf.to_int32(tf.round(value[:,:,:,0]*255))

    # gather
    cm = plt.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = cm(np.arange(256))[:,:3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value

def color_image(image, num_classes=19):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def load(sess, checkpoint_path, mask=[], prefix=""):
    def get_var_to_restore_list(ckpt_path, mask=[], prefix=""):
        """
        Get all the variable defined in a ckpt file and add them to the returned var_to_restore list. Allows for partially defined model to be restored fomr ckpt files.
        Args:
            ckpt_path: path to the ckpt model to be restored
            mask: list of layers to skip
            prefix: prefix string before the actual layer name in the graph definition
        """
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_dict = {}
        for v in variables:
            name = v.name[:-2]
            skip=False
            #check for skip
            for m in mask:
                if m in name:
                    skip=True
                    continue
            if not skip:
                variables_dict[v.name[:-2]] = v
        #print(variables_dict)
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        #print(var_to_shape_map)
        var_to_restore = {}
        for key in var_to_shape_map:
            #print(key)
            if prefix+key in variables_dict.keys():
                var_to_restore[key] = variables_dict[prefix+key]
        return var_to_restore
    print(" [*] Reading checkpoint...")
    if os.path.isdir(checkpoint_path):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt:
            c=True 
            model_checkpoint_path = ckpt.model_checkpoint_path            
        else:
            c= False
    else:
        c=True
        model_checkpoint_path = checkpoint_path
    
    if c and model_checkpoint_path:
        q = model_checkpoint_path.split("-")[-1]
        var_list=get_var_to_restore_list(model_checkpoint_path,mask=mask, prefix=prefix)
        print("Variable to restore: ", len(var_list))
        savvy = tf.train.Saver(var_list=var_list)
        savvy.restore(sess, model_checkpoint_path)
        return int(q) 
    else:
        return -1

def save(sess, saver, save_path, step):
    saver.save(sess,save_path,global_step=step)
