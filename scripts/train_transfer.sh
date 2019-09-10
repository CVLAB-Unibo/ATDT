data_path="/data/pier_data/CARLA/"
input_list="../filelist/input_list_train_carla.txt"
checkpoint_dir=" ../logs/transfer_depth_to_semantic_carla_to_cityscapes_original/"
target_task='semantic'
checkpoint_encoder_source="/home/pier/pier_data/CheckpointATDT/checkpoint_depth_mixed_carla_cityscapes_bn/ckpt"
checkpoint_target="/home/pier/pier_data/CheckpointATDT/checkpoint_semantic_carla_bn/ckpt"

python3 ../train_transfer.py --data_path $data_path --input_list $input_list --checkpoint_dir $checkpoint_dir --steps 100000 --batch_size 1 --lr 0.00001 --model dilated-resnet --target_task $target_task --normalizer_fn batch_norm --random_crop --crop_w 512 --crop_h 512 --checkpoint_encoder_source $checkpoint_encoder_source --checkpoint_encoder_target $checkpoint_target --checkpoint_decoder $checkpoint_target