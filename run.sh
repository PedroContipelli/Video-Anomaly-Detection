#!/usr/bin/env bash

#--------------------- C3D Experiments ----------------------------------#
#python3 main.py --train_classifier --gpu 0 --run_id run1_C3D --run_description "Run1 with loss for future prediction using C3D features." --model_init "None" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run2_C3D --run_description "Run2 with loss for future prediction using C3D features." --model_init "kaiming_normal" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run3_C3D --run_description "Run3 with loss for future prediction using C3D features." --model_init "kaiming_uniform" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run4_C3D --run_description "Run4 with loss for future prediction using C3D features." --model_init "xavier_normal" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run5_C3D --run_description "Run5 with loss for future prediction using C3D features." --model_init "xavier_uniform" --input_dim 4096 --features "c3d"
# NO TRANSFORMER ^^^

#--------------------- Transformer Experiments (C3D) ---------------------#
#python3 main.py --train_classifier --gpu 0 --run_id run6_C3D --run_description "Run6 with transformer using C3D features." --model_init "xavier_uniform" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run7_C3D --run_description "Run7 testing transformer bug" --model_init "None" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run8_C3D --run_description "Run8 testing transformer bug" --model_init "kaiming_normal" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run9_C3D --run_description "Run9 testing transformer bug" --model_init "kaiming_uniform" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run10_C3D --run_description "Run10 testing transformer bug" --model_init "xavier_normal" --input_dim 4096 --features "c3d"
#python3 main.py --train_classifier --gpu 0 --run_id run11_C3D --run_description "Run11 testing transformer bug" --model_init "xavier_uniform" --input_dim 4096 --features "c3d"



#--------------------- I3D Experiments ----------------------------------#
#python3 main.py --train_classifier --gpu 0 --run_id run1_I3D --run_description "Run1 with loss for future prediction using I3D features." --model_init "None" --input_dim 1024 --features "i3d"

#--------------------- R2Plus1D Experiments ----------------------------------#
#python3 main.py --train_classifier --gpu 0 --run_id run1_R2P1D --run_description "Run1 with loss for future prediction using R2P1D features." --model_init "None" --input_dim 512 --features "r2p1d"
#python3 main.py --train_classifier --gpu 0 --run_id run2_R2P1D --run_description "Run2 with loss for future prediction using R2P1D features. Using average pooled features from final conv layer." --model_init "None" --input_dim 4096 --features "r2p1d"
