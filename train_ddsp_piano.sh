#!/bin/bash

# Folder definition
maestro_path=$1
exp_dir=$2

# Constant training parameters over whole training
n_synths=16
steps_per_epoch=15904

# phase 1 training parameters
phase_1_batch_size=6
phase_1_n_epochs=TBA
phase_1_learning_rate=0.001

# phase 2 training parameters
phase_2_batch_size=3
phase_2_n_epochs=TBA
phase_2_learning_Rate=0.00001

python training_single_phase.py \
	--n_synths n_synths \
	--steps_per_epochs steps_per_epoch \
	--batch_size phase_1_batch_size \
	--n_epochs phase_1_n_epochs \
	--lr phase_1_learning_rate \
	maestro_path \
	exp_dir

python training_single_phase.py \
	--n_synths n_synths \
	--steps_per_epochs steps_per_epoch \
	--batch_size phase_2_batch_size \
	--n_epochs phase_2_n_epochs \
	--lr phase_2_learning_rate \
	--train_inharm True \
	--restore True \
	maestro_path \
	exp_dir

python training_single_phase.py \
	--n_synths n_synths \
	--steps_per_epochs steps_per_epoch \
	--batch_size phase_1_batch_size \
	--n_epochs phase_1_n_epochs \
	--lr phase_1_learning_rate \
	--restore True \
	maestro_path \
	exp_dir