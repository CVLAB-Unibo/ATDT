dataset_target='cityscapes'
data_path="/data/pier_data/CityScapes/"
input_list="../filelist/input_list_val_cityscapes.txt"
task='semantic'
test_dir="/data/pier_data/ATDT/test_codice/Table1-c2"
output=$test_dir'/results.txt'

python3 ../eval_task.py --dataset_target $dataset_target --data_path $data_path --input_list $input_list --task $task --pred_folder $test_dir/predictions --output $output --convert_to carla --convert_gt
