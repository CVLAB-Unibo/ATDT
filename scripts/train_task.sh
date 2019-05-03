data_path="/data/pier_data/CARLAv2/"
input_list="../filelist/input_list_train_carla.txt"
checkpoint_dir=" ../logs/semantic_carlav2/"
task='semantic'

python3 ../train_task.py --data_path $data_path --input_list $input_list --checkpoint_dir $checkpoint_dir --steps 25000 --batch_size 8 --task $task --normalizer_fn batch_norm --resize --resize_h 256 --resize_w 512 --crop --crop_w 128 --crop_h 128