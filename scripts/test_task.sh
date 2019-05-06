data_path="/data/pier_data/CARLAv2/"
input_list="../filelist/input_list_val_carla.txt"
checkpoint_dir=" ../logs/semantic_carlav2/"
task='semantic'
test_dir='../test/'$task'_carlav2'
python3 ../test_task.py --data_path $data_path --input_list $input_list --checkpoint_path $checkpoint_dir --task $task --normalizer_fn batch_norm --resize --resize_h 256 --resize_w 512 --save_predictions --test_dir $test_dir