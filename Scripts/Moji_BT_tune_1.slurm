#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Moji_BT_tune_1

# load required modules

module load gcccore/10.2.0
module load python/3.8.6
module load fosscuda/2020b
# module load pytorch/1.7.1-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0

cd /Fair_Data_Distillation

python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 1016129 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_0/
sleep 2
python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 45099 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_1/
sleep 2
python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 6906875 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_2/
sleep 2
python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 9526038 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_3/
sleep 2
python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 3714255 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_4/
sleep 2
python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 658918 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_5/
sleep 2
python main.py --mode distill_basic --dataset Moji --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --epochs 50 --lr 0.01 --ntoken 5000 --ninp 2304 --num_workers 0 --base_seed 9579580 --adv_debiasing False --BT Resampling --BTObj stratified_y --results_dir /experimental_results/Moji_BT_tune_1_6/
sleep 2
