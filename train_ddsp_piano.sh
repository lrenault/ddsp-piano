#!/bin/bash

# Folders definition
maestro_path=$1
exp_dir=$2

# Constant training parameters over whole training
# steps_per_epoch=15904
steps_per_epoch=4

# phase 1 training parameters
phase_1_batch_size=6
phase_1_n_epochs=4  # TBA
phase_1_learning_rate=0.001

# phase 2 training parameters
phase_2_batch_size=3
phase_2_n_epochs=4  # TBA
phase_2_learning_rate=0.00001

# phase 3 training parameters
phase_3_batch_size=6
phase_3_n_epochs=4  # TBA
phase_3_learning_rate=0.001

python training_single_phase.py \
	--steps_per_epoch $steps_per_epoch \
	--batch_size $phase_1_batch_size \
	--epochs $phase_1_n_epochs \
	--lr $phase_1_learning_rate \
	--phase 1 \
	$maestro_path \
	$exp_dir

python training_single_phase.py \
	--steps_per_epoch $steps_per_epoch \
	--batch_size $phase_2_batch_size \
	--epochs $phase_2_n_epochs \
	--lr $phase_2_learning_rate \
	--phase "2" \
	--restore "$exp_dir/phase_1/last_iter/" \
	$maestro_path \
	$exp_dir

python training_single_phase.py \
	--steps_per_epoch $steps_per_epoch \
	--batch_size $phase_3_batch_size \
	--epochs $phase_3_n_epochs \
	--lr $phase_3_learning_rate \
	--phase "3" \
	--restore "$exp_dir/phase_2/last_iter/" \
	$maestro_path \
	$exp_dir