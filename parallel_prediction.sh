#!/bin/bash

# Function to convert seconds to days:hours:minutes format
format_time() {
    local seconds=$1
    local days=$((seconds/86400))
    local hours=$(( (seconds%86400)/3600 ))
    local minutes=$(( (seconds%3600)/60 ))
    echo "${days}:${hours}:${minutes}"
}

# Record start time
start_time=$(date +%s)

# Run simulations in parallel
python cherry_blossom_prediction.py --sim_id 1 --dim_reservoir 5000 --seed 42 --prop_recurrent 0.1 --chaos_factor 1.2 --device cuda:0 &
python cherry_blossom_prediction.py --sim_id 2 --dim_reservoir 5000 --seed 42 --prop_recurrent 0.2 --chaos_factor 1.2 --device cuda:1 &
python cherry_blossom_prediction.py --sim_id 3 --dim_reservoir 5000 --seed 42 --prop_recurrent 0.1 --chaos_factor 1.5 --device cuda:0 &
python cherry_blossom_prediction.py --sim_id 4 --dim_reservoir 5000 --seed 42 --prop_recurrent 0.2 --chaos_factor 1.5 --device cuda:1

# Wait for all processes to finish
wait

# Start next set of simulations in parallel
python cherry_blossom_prediction.py --sim_id 1 --dim_reservoir 10000 --seed 42 --prop_recurrent 0.1 --chaos_factor 1.2 --device cuda:0 &
python cherry_blossom_prediction.py --sim_id 2 --dim_reservoir 10000 --seed 42 --prop_recurrent 0.2 --chaos_factor 1.2 --device cuda:1 &
python cherry_blossom_prediction.py --sim_id 3 --dim_reservoir 10000 --seed 42 --prop_recurrent 0.1 --chaos_factor 1.5 --device cuda:0 &
python cherry_blossom_prediction.py --sim_id 4 --dim_reservoir 10000 --seed 42 --prop_recurrent 0.2 --chaos_factor 1.5 --device cuda:1

wait

# Record end time
parallel_end_time=$(date +%s)
parallel_duration=$((parallel_end_time - start_time))
parallel_runtime=$(format_time $parallel_duration)

echo "Parallel run completed in $parallel_runtime"