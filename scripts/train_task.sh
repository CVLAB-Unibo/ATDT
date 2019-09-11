data_path="/data/pier_data/"
input_list="../filelist/input_list_train_mixed_carlav2_cityscapes.txt"
checkpoint_dir=" ../logs/depth_mixed_carlav2_cityscapes/"
task='depth'

python3 ../train_task.py --data_path $data_path --input_list $input_list --checkpoint_dir $checkpoint_dir --steps 150000 --batch_size 8 --task $task --normalizer_fn batch_norm --crop --crop_w 512 --crop_h 512