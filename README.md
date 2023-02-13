# Fair_Dataset_Distillation
Source codes for SustaiNLP@EMNLP 2022 paper "Towards Fair Supervised Dataset Distillation for Text Classification"

```
@inproceedings{han-etal-2022-towards-fair,
    title = "Towards Fair Dataset Distillation for Text Classification",
    author = "Han, Xudong  and
      Shen, Aili  and
      Li, Yitong  and
      Frermann, Lea  and
      Baldwin, Timothy  and
      Cohn, Trevor",
    booktitle = "Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sustainlp-1.13",
    pages = "65--72",
    abstract = "With the growing prevalence of large-scale language models, their energy footprint and potential to learn and amplify historical biases are two pressing challenges. Dataset distillation (DD) {---} a method for reducing the dataset size by learning a small number of synthetic samples which encode the information in the original dataset {---} is a method for reducing the cost of model training, however its impact on fairness has not been studied. We investigate how DD impacts on group bias, with experiments over two language classification tasks, concluding that vanilla DD preserves the bias of the dataset. We then show how existing debiasing methods can be combined with DD to produce models that are fair and accurate, at reduced training cost.",
}

```

## Overview

In this work, we first show that dataset distillation preserves the bias of the dataset, and then propose a framework to combine existing debiaisng methods to produce models that are fair and accurate, at reduced training cost. 


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
