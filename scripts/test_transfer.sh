data_path="/data/pier_data/CityScapes/"
input_list="../filelist/input_list_val_cityscapes.txt"
checkpoint_dir=" ../logs/transfer_depth_to_semantic_carla_to_cityscapes_original/"
test_dir="../transfer_depth_to_semantic_carla_to_cityscapes_original/"
checkpoint_encoder_source="/home/pier/pier_data/CheckpointATDT/checkpoint_depth_mixed_carla_cityscapes_bn/ckpt"
checkpoint_target="/home/pier/pier_data/CheckpointATDT/checkpoint_semantic_carla_bn/ckpt"

python3 ../test_transfer.py --data_path $data_path --input_list $input_list --checkpoint_dir $checkpoint_dir --test_dir $test_dir --normalizer_fn batch_norm --model dilated-resnet --checkpoint_encoder_source $checkpoint_encoder_source --checkpoint_decoder $checkpoint_target --save_predictions --target_task semantic