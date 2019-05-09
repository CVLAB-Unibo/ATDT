data_path="/data/pier_data/CityScapes/"
input_list="../filelist/input_list_val_cityscapes.txt"
checkpoint_dir="../logs/transfer_dep2sem_carlav2/"
test_dir="../test/transfer_dep2sem_carla_to_cityscapes/"
checkpoint_encoder_source="../logs/depth_carlav2/depth_dilated-resnet-25000"
checkpoint_decoder="../logs/semantic_carlav2/semantic_dilated-resnet-25000"

python3 ../test_transfer.py --data_path $data_path --input_list $input_list --checkpoint_dir $checkpoint_dir --test_dir $test_dir --normalizer_fn batch_norm --model dilated-resnet --checkpoint_encoder_source $checkpoint_encoder_source --checkpoint_decoder $checkpoint_decoder --resize --resize_w 512 --resize_h 256 --save_predictions --target_task semantic