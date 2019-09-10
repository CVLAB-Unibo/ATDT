data_path="/data/pier_data/CityScapes/"
input_list="../filelist/input_list_val_cityscapes.txt"
checkpoint_dir=" ../logs/depth_mixed_carlav2_cityscapes_cityscapes/"
task='depth'
test_dir='../test/'$task'_carlav2_to_cityscapes_original'
python3 ../test_task.py --data_path $data_path --input_list $input_list --checkpoint_path $checkpoint_dir --task $task --normalizer_fn batch_norm --save_predictions --test_dir $test_dir
