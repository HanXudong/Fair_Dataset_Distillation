# Fair_Dataset_Distillation
Source codes for SustaiNLP@EMNLP 2022 paper "Towards Fair Supervised Dataset Distillation for Text Classification"

Xudong Han, Aili Shen, Yitong Li, Lea Frermann, Timothy Baldwin, and Trevor Cohn. 2022. Towards Fair Dataset Distillation for Text Classification. In Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP), pages 65–72, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

## Overview

In this work, we first show that although dataset distillation preserves the bias of the dataset, and then propose a framework to combine existing debiaisng methods to produce models that are fair and accurate, at reduced training cost. 


## Code
This dir includes source codes for reproducing our experimental results in paper. 

- The text distillation is deployed based on https://github.com/ilia10000/dataset-distillation
- The bias mitigation approaches are based on https://github.com/HanXudong/fairlib

### Accessing Fairness 
Fairness evaluation metrics are included in [Fair_Dataset_Distillation/fairness_src/evaluator](https://github.com/HanXudong/Fair_Dataset_Distillation/tree/main/fairness_src/evaluator).

Since additional protected labels are required for fairness evaluation and bias mitigation, we provide example dataloaders in [Fair_Dataset_Distillation/fairness_src/dataloaders/](https://github.com/HanXudong/Fair_Dataset_Distillation/tree/main/fairness_src/dataloaders).

### Preprocessing

Similar to the implementation of fairlib, preprocessing approaches are combined with the [BaseDataset class](https://github.com/HanXudong/Fair_Dataset_Distillation/blob/main/datasets/utils.py), where the distributions of target labels and demographics are balanced. 

### In-processing

Adversarial training and fair contrastive learning are implemented in [Fair_Dataset_Distillation/fairness_src/networks/](https://github.com/HanXudong/Fair_Dataset_Distillation/tree/main/fairness_src/networks).

The inclusion of in-processing methods aims at learning fairer synthetic datasets, which can be seen from [here](https://github.com/HanXudong/Fair_Dataset_Distillation/blob/e6db24bde81db038872e753631c1d49963c12c73/train_distilled_image.py#L158-L178).


## Scripts

- To reproduce experimental results in this paper, please see the scripts in the `Fair_Dataset_Distillation\Scripts`. 

The scripts name is in the following format:
`{Dataset}_{Method}_tune_{Number of instances per class}.slurm`

Within each file, you should be able to find corresponding command line for running code with all required hyperparameters, for example,

```
python main.py --mode distill_basic --dataset Bios --arch MLPClassifier --distill_steps 1 --train_nets_type known_init --n_nets 1 --test_nets_type same_as_train --static_labels 0 --random_init_labels zeros --textdata True --fairness True --distill_epochs 3 --distill_lr 0.01 --decay_epochs 10 --epochs 30 --lr 0.01 --ntoken 5000 --ninp 768 --num_workers 0 --base_seed 88613 --adv_debiasing False --results_dir /experimental_results/Bios_Vanilla_tune_1_0/
```

## Dataset

Please follow the instructions [https://github.com/HanXudong/fairlib](https://github.com/HanXudong/fairlib/tree/main/data).