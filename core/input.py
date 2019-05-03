from __future__ import absolute_import, division, print_function
import tensorflow as tf
import cv2
import os
import numpy as np
import abc

class MetaLoader():
    """
    Meta Class to share code between different data loader
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass
    
    def _decode_filelist(self):
        self._input_queue = tf.train.string_input_producer([self.filenames_file], shuffle=self._shuffle)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(self._input_queue)
        return tf.string_split([line],delimiter=";").values
    
    def string_length_tf(self,t):
        return tf.py_func(len, [t], [tf.int64])

class FeatureLoader(MetaLoader):
    def __init__(self, data_path,filenames_file,feature_shape=[32,32,2048],batch_size=4,mode='train',shuffle=False,num_threads=4):
        self.data_path = data_path
        self.filenames_file = filenames_file
        self.batch_size=batch_size
        self._shuffle=shuffle
        self._num_threads=num_threads
        self._feature_shape=feature_shape
        self.mode=mode
        self._load_inputs()

    def read_npz(file_name,key='arr_0'):
        file_name=file_name.decode('utf-8')
        t = np.load(file_name)
        return t['arr_0']

    def _laod_np(self,path_op,shape):
        data = tf.py_func(lambda x: read_npz(x),[path_op],tf.float32)
        data = tf.reshape(data,shape)
        return data

    def _load_inputs(self):
        
        tokens = self._decode_filelist()

        domain_a_path = tf.string_join([self.data_path,tokens[0]])
        self.domain_a=self._laod_np(domain_a_path,self._feature_shape)

        domain_b_path = tf.string_join([self.data_path,tokens[1]])
        self.domain_b=self._laod_np(domain_b_path,self._feature_shape)

        if self.mode == 'train':
            self._batch_a, self._batch_b = tf.train.batch([self.domain_a,self.domain_b],self.batch_size,capacity=self.batch_size*50, num_threads=self._num_threads)
            self.inputs = (self._batch_a,self._batch_b)
        if self.mode == 'test':
            self.inputs = (tf.expand_dims(self.domain_a,axis=0),tf.expand_dims(self.domain_b,axis=0))

class Dataloader(MetaLoader):

    def __init__(self, params, noShuffle=False, central_crop=False):
        self.data_path = params.data_path
        self.mode = params.mode
        self.filenames_file = params.filenames_file
        self.type= 'uint8'#cv2.imread(os.path.join(self.data_path, open(self.filenames_file).readlines[0].split(";")[3]),cv2.IMREAD_ANYDEPTH).dtype
        self.task = params.task
        
        self.resize = params.resize
        if self.resize:
            self.resize_h = params.resize_h
            self.resize_w = params.resize_w
        self.crop = params.crop
        self.central_crop = central_crop
        if self.crop:
            self.crop_h = params.crop_h
            self.crop_w = params.crop_w
        self.batch_size = params.batch_size

        self._shuffle=(not noShuffle)

        self._load_inputs()
    
    def _load_inputs(self):
        split_line = self._decode_filelist()

        image_path  = tf.string_join([self.data_path, split_line[0]])
        image_o  = self.read_image(image_path)[:,:,:3]
        
        
        if self.task == 'semantic' or self.task =='semantic-aligned':
            semantic_image_path = tf.string_join([self.data_path, split_line[2]])
            gt = self.read_semantic_gt(semantic_image_path)
            gt.set_shape([None, None, 1])
        elif self.task == 'depth' or self.task =='depth-aligned':
            depth_image_path = tf.string_join([self.data_path, split_line[3]])
            gt  = self.read_depth(depth_image_path)
            gt.set_shape([None, None, 1])
        elif self.task == 'unsupervised-depth':
            gt = tf.string_join([self.data_path, split_line[1]])
            gt  = self.read_image(image_right_path)[:,:,:3]
        elif self.task == 'normals':
            normals_path = tf.string_join([self.data_path, split_line[4]])
            gt  = self.read_normals(normals_path)
            gt.set_shape([None, None, 3])

        if self.mode == 'train':
            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            if self.task == 'unsupervised-depth':
                image, gt = tf.cond(do_augment > 0.5, lambda: self.augment_image(image_o, gt), lambda: (image_o, gt))
                image_right.set_shape([None, None, 3])
            else:
                image = tf.cond(do_augment > 0.5, lambda: self.augment_image(image_o), lambda: (image_o))
            
            
            # resizing
            if self.resize:
                image=tf.image.resize_images(image,[self.resize_h,self.resize_w])

                if self.task == 'semantic' or self.task =='semantic-aligned':
                    gt = tf.image.resize_images(gt,[self.resize_h,self.resize_w],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                else:
                    gt = tf.image.resize_images(gt,[self.resize_h,self.resize_w])
            
            if self.crop:
                if not self.central_crop:
                    print("Random Crop")
                    crop_offset_w = tf.cond(tf.equal(tf.shape(image)[1]- self.crop_w,0), lambda : 0, lambda : tf.random_uniform((), minval=0, maxval= tf.shape(image)[1]- self.crop_w, dtype=tf.int32))
                    crop_offset_h = tf.cond(tf.equal(tf.shape(image)[0]- self.crop_h,0), lambda : 0, lambda : tf.random_uniform((), minval=0, maxval= tf.shape(image)[0]- self.crop_h, dtype=tf.int32))
                    image = tf.image.crop_to_bounding_box(image, crop_offset_h, crop_offset_w, self.crop_h, self.crop_w)
                    gt = tf.image.crop_to_bounding_box(gt, crop_offset_h, crop_offset_w, self.crop_h, self.crop_w)
                else:
                    print("Central Crop")
                    image = tf.image.resize_image_with_crop_or_pad(image, self.crop_h, self.crop_w)
                    gt = tf.image.resize_image_with_crop_or_pad(gt, self.crop_h, self.crop_w)
               
            image.set_shape([None, None, 3])

            min_after_dequeue = 50
            capacity = 500
            num_threads = 8
            
            if not self._shuffle:
                self.inputs = tf.train.batch([image, gt], self.batch_size, capacity)
            else:
                self.inputs = tf.train.shuffle_batch([image, gt], self.batch_size, capacity, min_after_dequeue, num_threads)

        elif self.mode == 'test':
            self.image_batch = image_o
            self.image_batch.set_shape([None,None, 3])
            if self.crop:
                self.image_batch = tf.image.resize_image_with_crop_or_pad(self.image_batch,self.crop_h,self.crop_w)
                self.image_batch.set_shape([self.crop_h,self.crop_w, 3])
            self.image_batch = tf.expand_dims(self.image_batch,axis=0)
            self.inputs = self.image_batch

    def augment_image(self, image, right_image=None):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        image_aug  = image  ** random_gamma
        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_aug  =  image_aug * random_brightness
        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        image_aug  *= color_image
        # saturate
        image_aug  = tf.clip_by_value(image_aug,  0, 1)
        if right_image is None:
            return image_aug  
        else:
            right_image_aug = right_image  ** random_gamma
            right_image_aug  =  right_image_aug * random_brightness
            right_image_aug  *= color_image
            return image_aug, right_image_aug
        
    def read_semantic_gt(self, image_path):
        image  = tf.image.decode_png(tf.read_file(image_path))
        return image

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = self.string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), 
            lambda: tf.image.decode_png(tf.read_file(image_path)))
        image  = tf.image.convert_image_dtype(image,  tf.float32)
        return image
    
    def read_depth(self,image_path):
        path_length = self.string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'exr')
        image = tf.cond(file_cond, lambda: self.decode_exr(image_path), lambda: self.read_png(image_path))
        return image
    
    def read_normals(self,image_path):
        normals_rgb=tf.cast(tf.image.decode_png(tf.read_file(image_path)),tf.float32)
        normals=(normals_rgb/255*2)-1
        return normals

    def read_png(self, image_path):
        if self.type=='uint16':
            return self.decode_depth_16bit(tf.read_file(image_path))
        else:
            return self.decode_depth_3_channels(tf.image.decode_png(tf.read_file(image_path)))
    
    def decode_exr(self,filepath):
        def readexr(filepath):
            f=100 #depth in m
            return np.expand_dims(cv2.imread(filepath.decode(),cv2.IMREAD_UNCHANGED)[:,:,-1]*f,axis=-1)
        return tf.py_func(readexr,[filepath],tf.float32)

    def decode_depth_3_channels(self,raw):
        raw = tf.cast(raw[:,:,:3],tf.float32)
        f= 1000 #depth in m
        out = raw[:,:,0] + tf.multiply(raw[:,:,1],256) + tf.multiply(raw[:,:,2],256*256)
        out = tf.divide(out , (256*256*256 - 1))
        out = tf.expand_dims(out*f,axis=-1)
        return out

    def decode_depth_16bit(self,raw):
        out = tf.image.decode_png(raw,tf.uint16)
        out=tf.cast(out,tf.float32)
        out=out/100 #depth in m
        return out