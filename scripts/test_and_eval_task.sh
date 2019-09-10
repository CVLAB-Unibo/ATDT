dataset_target='cityscapes'
data_path="/data/pier_data/CityScapes/"
input_list="../filelist/input_list_val_cityscapes.txt"
checkpoint_dir=" ../logs/semantic_carlav2_original/"
task='semantic'
test_dir='../test/'$task'_carlav2_original_butresized'
#python3 ../test_task.py --data_path $data_path --input_list $input_list --checkpoint_path $checkpoint_dir --task $task --normalizer_fn batch_norm --save_predictions --test_dir $test_dir --resize --resize_h 256 --resize_w 512
##EVAL
test_dir=$test_dir'/predictions'
output=$test_dir'/results.txt'
python3 ../eval_task.py --dataset_target $dataset_target --data_path $data_path --input_list $input_list --task $task --pred_folder $test_dir --output $output --convert_to carla --convert_gt --resize --resize_h 256 --resize_w 512