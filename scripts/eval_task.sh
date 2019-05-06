dataset_target='carla'
data_path="/data/pier_data/CARLAv2/"
input_list="../filelist/input_list_val_carla.txt"
task='semantic'
test_dir='/data/pier_data/ATDT/test/semantic_carlav2/predictions/'
output='/data/pier_data/ATDT/test/semantic_carlav2/results.txt'

python3 ../eval_task.py --dataset_target $dataset_target --data_path $data_path --input_list $input_list --task $task --resize --resize_h 256 --resize_w 512 --pred_folder $test_dir --output $output