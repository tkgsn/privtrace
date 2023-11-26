# python3 main.py --total_epsilon 1
# python3 main.py --total_epsilon 5
# python3 main.py --total_epsilon 10
# python3 main.py --total_epsilon 100
# python3 main.py --total_epsilon 1000

# python3 data_pre_processing.py  --dataset taxi --data_name taxi_100000 --save_name privtrace --dataset_config taxi.json
# python3 main.py --total_epsilon 1 --fixed_divide_parameter 20 --dataset taxi --data_name taxi_100000 --training_data_name privtrace --save_name privtrace_epsilon1
# python3 main.py --total_epsilon 3 --fixed_divide_parameter 20 --dataset taxi --data_name taxi_100000 --training_data_name privtrace --save_name privtrace_epsilon3
# python3 main.py --total_epsilon 5 --fixed_divide_parameter 20 --dataset taxi --data_name taxi_100000 --training_data_name privtrace --save_name privtrace_epsilon5
# python3 main.py --total_epsilon 0.15 --fixed_divide_parameter 20 --dataset taxi --data_name taxi_100000 --training_data_name privtrace --save_name privtrace_epsilon0.15


#!/bin/bash

cd ../priv_traj_gen

dataset=geolife_mm
max_size=0
data_name=${max_size}
latlon_config=geolife_mm.json
location_threshold=200
time_threshold=30
n_bins=30
seed_for_dataset=0
training_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed_for_dataset}

# python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name --n_bins $n_bins
# python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins $option --seed $seed_for_dataset

# location_threshold=0
# time_threshold=0
# route_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed_for_dataset}

# python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $route_data_name --n_bins $n_bins $option --seed $seed_for_dataset

cd ../privtrace

total_epsilon=1
# save_name=privtrace_seed${seed_for_dataset}_eps$total_epsilon
save_name=200_30_bin30_seed0
dataset_config_path=../priv_traj_gen/config.json
python3 make_training_data.py --dataset $dataset --data_name $data_name --save_name $save_name --dataset_config $dataset_config_path --dataset_seed $seed_for_dataset
n_bins=30
python3 privtrace_generator.py --dataset $dataset --data_name $data_name --training_data_name $save_name --dataset_config_path $dataset_config_path --n_bins $n_bins --total_epsilon $total_epsilon

evaluate_stay_traj_dir=/data/$dataset/$data_name/$training_data_name
evaluate_route_traj_dir=/data/$dataset/$data_name/$route_data_name
save_dir=/data/results/$dataset/$data_name/$save_name

python3 evaluate.py $evaluate_stay_traj_dir $evaluate_route_traj_dir $save_dir