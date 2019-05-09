data_path="/data/pier_data/CARLAv2/"
input_list="../filelist/input_list_train_carla.txt"
checkpoint_dir=" ../logs/transfer_dep2sem_carlav2/"
target_task='semantic'
checkpoint_encoder_source='/data/pier_data/ATDT/logs/depth_carlav2/depth_dilated-resnet-25000'
checkpoint_encoder_target='/data/pier_data/ATDT/logs/semantic_carlav2/semantic_dilated-resnet-25000'

python3 ../train_transfer.py --data_path $data_path --input_list $input_list --checkpoint_dir $checkpoint_dir --steps 100000 --batch_size 1 --lr 0.00001 --model dilated-resnet --target_task $target_task --normalizer_fn batch_norm --resize --resize_h 256 --resize_w 512 --random_crop --crop_w 128 --crop_h 128 --checkpoint_encoder_source $checkpoint_encoder_source --checkpoint_encoder_target $checkpoint_encoder_target --checkpoint_decoder $checkpoint_encoder_target 