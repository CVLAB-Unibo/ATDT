### PARAMS
crop_h=512
crop_w=512

source_task='depth'
target_task='semantic'

dataset_source='carla'
path_dataset_source="/data/pier_data/CARLA"
path_source_list="../filelist/input_list_train_carla.txt"

dataset_target='cityscapes'
path_target_dataset="/data/pier_data/CityScapes/"
path_target_list="../filelist/input_list_train_cityscapes.txt"
path_target_validation_list="../filelist/input_list_val_cityscapes.txt"

path_mixed="/data/pier_data/"
path_mixed_list="../filelist/input_list_train_mixed_carla_cityscapes.txt"

config="A_"$dataset_source"_B_"$dataset_target"_1_"$source_task"_2_"$target_task
default_dir=$"../runs/"$config

if [ ! -e $default_dir ]; then
    mkdir $default_dir
fi

### SOURCE NETWORK
path_checkpoint_task_source=$default_dir"/"$source_task"_"$dataset_source"_and_"$dataset_target

### TARGET NETWORK
path_checkpoint_task_target=$default_dir"/"$target_task"_"$dataset_source

###TRANSFER
path_checkpoint_transfer=$default_dir"/"$source_task"_to_"$target_task"_"$dataset_source

### EVALUATION
### BASELINE
test_dir_baseline=$default_dir"/baseline"
path_results_baseline=$test_dir_baseline'/results.txt'

### TRANSFER
test_dir_transfer=$default_dir"/ATDT"
path_results_transfer=$test_dir_transfer'/results.txt'

python3 ../train_task.py --data_path $path_mixed --input_list $path_mixed_list --checkpoint_dir $path_checkpoint_task_source --steps 150000 --batch_size 8 --task $source_task --normalizer_fn batch_norm --crop --crop_w $crop_w --crop_h $crop_h

python3 ../train_task.py --data_path $path_dataset_source --input_list $path_source_list --checkpoint_dir $path_checkpoint_task_target --steps 25000 --batch_size 8 --task $target_task --normalizer_fn batch_norm --crop --crop_w $crop_w --crop_h $crop_h

python3 ../train_transfer.py --data_path $path_dataset_source --input_list $path_source_list --checkpoint_dir $path_checkpoint_transfer --checkpoint_encoder_source $path_checkpoint_task_source --checkpoint_encoder_target $path_checkpoint_task_target --checkpoint_decoder $path_checkpoint_task_target --steps 100000 --batch_size 1 --lr 0.00001 --model dilated-resnet --target_task $target_task --normalizer_fn batch_norm --random_crop --crop_w $crop_w --crop_h $crop_h

python3 ../test_task.py --data_path $path_target_dataset --input_list $path_target_validation_list --checkpoint_path $path_checkpoint_task_target --task $target_task --normalizer_fn batch_norm --save_predictions --test_dir $test_dir_baseline
python3 ../eval_task.py --dataset_target $dataset_target --data_path $path_target_dataset --input_list $path_target_validation_list --task $target_task --pred_folder $test_dir_baseline/predictions --output $path_results_baseline --convert_to carla --convert_gt

python3 ../test_transfer.py --data_path $path_target_dataset --input_list $path_target_validation_list --checkpoint_dir $path_checkpoint_transfer --checkpoint_encoder_source $path_checkpoint_task_source --checkpoint_decoder $path_checkpoint_task_target --test_dir $test_dir_transfer --normalizer_fn batch_norm --model dilated-resnet  --save_predictions --target_task $target_task
python3 ../eval_task.py --dataset_target $dataset_target --data_path $path_target_dataset --input_list $path_target_validation_list --task $target_task --pred_folder $test_dir_transfer/predictions --output $path_results_transfer --convert_to carla --convert_gt
