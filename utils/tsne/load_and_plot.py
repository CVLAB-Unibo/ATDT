import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import cv2
from tensorflow.contrib.tensorboard.plugins import projector


SPRITE_FILENAME_RGB = 'sprite_rgb.png'
SPRITE_FILENAME_PRED = 'sprite_pred.png'
METADATA_FILENAME = 'metadata.tsv'
NAMEMAP_FILENAME = 'namemap.tsv'
EMBEDDING_FILENAME = 'embedding.ckpt'
EMBEDDING_NPY = 'embedding.npy'
MAX_SPRITE_SIDE = 8192
SPRITE_SHAPE = [32,32,3]

if __name__=='__main__':
    #create args
    parser = argparse.ArgumentParser(description='script to load different numpy array and create suitable files to project them all together')
    parser.add_argument('--folders', help="path to the projector folders that we want to mix together", nargs='+', required=True)
    parser.add_argument('--out_dir', help="path to the output folder where the result will be saved", required=True)

    args = parser.parse_args()

    #load numpy arrays 
    tensors = []
    metadata_lines = []
    sprites = []
    for i,f in enumerate(args.folders):
        folder_name = os.path.basename(f)
        assert(os.path.isdir(f))
        
        #load numpy array
        numpy_path = os.path.join(f,EMBEDDING_NPY)
        assert(os.path.exists(numpy_path))
        tensor = np.load(numpy_path)
        tensors.append(tensor)

        #load metadata
        metadata_path = os.path.join(f,METADATA_FILENAME)
        assert(os.path.exists(metadata_path))
        with open(metadata_path) as f_in:
            lines = f_in.readlines()
        if i==0:
            new_lines = [lines[0].strip()+'\tTest Id\tTest+Dataset\n']
        else:
            new_lines = []

        for l in lines[1:]:
            dataset_id = int(l.split('\t')[1])
            test_id = 10*i + dataset_id
            new_lines.append(l.strip() + '\t{}\t{}\n'.format(f,test_id))
        metadata_lines += new_lines

        #load sprites
        sprite_path = os.path.join(f,SPRITE_FILENAME_RGB)
        assert(os.path.exists(sprite_path))
        sprite=cv2.imread(sprite_path,-1)
        sprites.append(sprite)

    #concat them all togetehr
    tensor_to_save = np.concatenate(tensors, axis=0)
    sprite_to_save = np.concatenate(sprites, axis=0)
    
    #create output dirs
    os.makedirs(args.out_dir,exist_ok=True)
    with tf.Session() as sess:
        
        #create embedding var inside tf and initialize it
        with tf.variable_scope('embedding'):
            encoding = tf.Variable(tensor_to_save, trainable=False)
        sess.run(encoding.initializer)
        print("Initialized variable")

        #save metadata
        meta_out = os.path.join(args.out_dir,METADATA_FILENAME)
        with open(meta_out,'w+') as f_out:
            f_out.writelines(metadata_lines)
        
        #save sprite
        sprite_out = os.path.join(args.out_dir,SPRITE_FILENAME_RGB)
        cv2.imwrite(sprite_out,sprite_to_save)
        
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

