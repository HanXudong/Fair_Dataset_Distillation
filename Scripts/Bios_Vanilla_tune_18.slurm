#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Bios_Vanilla_tune_18

# load required modules

module load gcccore/10.2.0
module load python/3.8.6
module load fosscuda/2020b
# module load pytorch/1.7.1-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0

cd /Fair_Data_Distillation

python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 2603733 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_0/
sleep 2
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 1789696 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_1/
sleep 2
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 4328370 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_2/
sleep 2
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 4008957 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_3/
sleep 2
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 2120780 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_4/
sleep 2
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 6106072 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_5/
sleep 2
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 18 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 8431760 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_18_6/
sleep 2
