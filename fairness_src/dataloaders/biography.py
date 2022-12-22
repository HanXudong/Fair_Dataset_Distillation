from typing import Dict
import numpy as np
import pickle
from numpy.lib.function_base import _parse_input_dimensions

import torch
import torch.utils.data as data
from pathlib import Path, PureWindowsPath
from collections import defaultdict, Counter

import pandas as pd
from random import shuffle

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

# Dataloader for the full bios dataset
class BiosDataset(torch.utils.data.Dataset):
    def __init__(self, 
                data_dir, 
                split, 
                embedding_type = "bert_avg_SE",
                protected_task = "gender",
                weighting = None,
                balance_type = None,
                full_label = True
                ):
                
        self.data_dir = data_dir
        
        # check split
        self.dataset_type = {"train", "dev", "test"}
        assert split in self.dataset_type, "split should be one of train, dev, and test."
        self.split = split
        self.filename = "bios_{}_df.pkl".format(split)
        
        # check embedding type
        assert embedding_type in ("bert_avg_SE", "bert_cls_SE"), "Embedding should either be avg or cls."
        """
            avg and cls are sentence representations from BERT encoders
        """
        self.embedding_type = embedding_type

        # check protected task
        assert protected_task in ["gender", "economy", "both"], "Bios dataset supports only single and intersectional."
        self.protected_task = protected_task

        # save the state if instances need to be labelled with all protected attribtues
        self.full_label = full_label

        # Init 
        self.X = []
        self.y = []
        self.protected_label = []
        
        # return weights
        self.weighting = weighting

        # Load preprocessed data
        print("Loading data")
        self.load_data()

        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y)
        self.protected_label = np.array(self.protected_label)
        # self.location_label = np.array(self.location_label)

        if weighting is not None:
            assert weighting in ["ratio", "y", "g", "class", "class_g"], "not implemented"
            """
            reweighting each training instance 
                ratio:      y,g combination, p(g,y)
                y:          main task label y only, p(y)
                g:          protected label g only, p(g)
                class:      balancing the g for each y, p(g|y)
                class_g:    balancing the y for each g, p(y|g)
            """

            ## calculate weight for each instance
            
            # count the times of co-occurence of the target label and private label

            n_total = len(self.y)
            if weighting == "ratio":
                weighting_counter = Counter([(i,j) for i,j in zip(self.y, self.protected_label)])
            elif weighting == "y":
                weighting_counter = Counter(self.y)
            elif weighting == "g":
                weighting_counter = Counter(self.protected_label)
            elif weighting == "class":
                weighting_counter = Counter([(i,j) for i,j in zip(self.y, self.protected_label)])
            elif weighting == "class_g":
                weighting_counter = Counter([(i,j) for i,j in zip(self.y, self.protected_label)])
            else:
                pass
            
            if weighting in ["ratio", "y", "g"]:
                n_perfect_balanced = n_total / len(weighting_counter.keys())
                for k in weighting_counter.keys():
                    weighting_counter[k] = n_perfect_balanced / weighting_counter[k]
            elif weighting == "class":
                for k in weighting_counter.keys():
                    _y, _g = k
                    # modify the code to multiple groups
                    # weighting_counter[k] = (weighting_counter[(_y, 0)]+weighting_counter[(_y, 1)]) / 2.0 / weighting_counter[k]

                    groups_with_same_y = [_k for _k in weighting_counter.keys() if _k[0] == _y]
                    num_y = sum([weighting_counter[_k] for _k in groups_with_same_y])
                    weighting_counter[k] = num_y / len(groups_with_same_y) / weighting_counter[k]
            elif weighting == "class_g":
                for k in weighting_counter.keys():
                    _y, _g = k
                    groups_with_same_g = [_k for _k in weighting_counter.keys() if _k[1] == _g]
                    num_g = sum([weighting_counter[_k] for _k in groups_with_same_g])
                    weighting_counter[k] = num_g / len(groups_with_same_g) / weighting_counter[k]
            else:
                pass
            
            # add weights
            self.instance_weights = []
            for _y, _g in zip(self.y, self.protected_label):
                if weighting == "ratio":
                    self.instance_weights.append(weighting_counter[(_y, _g)])
                elif weighting == "y":
                    self.instance_weights.append(weighting_counter[_y])
                elif weighting == "g":
                    self.instance_weights.append(weighting_counter[_g])
                elif weighting == "class":
                    self.instance_weights.append(weighting_counter[(_y, _g)])
                elif weighting == "class_g":
                    self.instance_weights.append(weighting_counter[(_y, _g)])
                else:
                    pass
        
        if balance_type is not None:
            # check if weighting is None
            assert weighting is None, "resampling and reweighting can not be used at the same time."

            assert balance_type in ["y", "g", "ratio", "class", "stratified", "stratified_g"], "not implemented yet"
            """
                ratio:  according to its y,g combination, P(y,g)
                y:      according to its main task label y only, P(y)
                g:      according to its protected label g only, P(g)
                class:  according to its protected label within its main task label, P(g|y)
                stratified: keep the y distribution and balance g within y
                stratified_g: keep the g distribution and balance y within g
            """
            
            # init a dict for storing the index of each group.

            group_idx = {}
            if balance_type == "ratio":
                group_labels = [(i,j) for i,j in zip(self.y, self.protected_label)]
            elif balance_type == "y":
                group_labels = self.y
            elif balance_type == "g":
                group_labels = self.protected_label
            elif balance_type == "class":
                group_labels = [(i,j) for i,j in zip(self.y, self.protected_label)]
            elif balance_type == "stratified":
                group_labels = [(i,j) for i,j in zip(self.y, self.protected_label)]
            elif balance_type == "stratified_g":
                group_labels = [(j,i) for i,j in zip(self.y, self.protected_label)]
            else:
                pass

            for idx, group_label in enumerate(group_labels):
                group_idx[group_label] = group_idx.get(group_label, []) + [idx]

            selected_index = []
            if balance_type in ["ratio", "y", "g"]:
                # selected = min(len(man_idx), len(woman_idx))
                selected = min([len(i) for i in group_idx.values()])
                
                for index in group_idx.values():
                    _index = index
                    shuffle(_index)
                    selected_index = selected_index + _index[:selected]
            elif balance_type == "class":
                # balance protected groups with respect to each main task class

                # iterate each main task class
                for y in set(self.y):
                    # balance the protected group distribution
                    y_group_idx = [group_idx[(y, g)] for g in set(self.protected_label)]
                    y_selected = min([len(i) for i in y_group_idx])
                    for index in y_group_idx:
                        _index = index
                        shuffle(_index)
                        selected_index = selected_index + _index[:y_selected]
            elif balance_type == "stratified":
                # empirical distribution of y
                weighting_counter = Counter(self.y)

                # a list of (weights, actual length)
                condidate_selected = min([len(group_idx[k])/weighting_counter[k[0]] for k in group_idx.keys()])

                distinct_y_label = set(self.y)
                distinct_g_label = set(self.protected_label)
                
                # iterate each main task class
                for y in distinct_y_label:
                    selected = int(condidate_selected * weighting_counter[y])
                    for g in distinct_g_label:
                        _index = group_idx[(y,g)]
                        shuffle(_index)
                        selected_index = selected_index + _index[:selected]
            elif balance_type == "stratified_g":
                # empirical distribution of g
                weighting_counter = Counter(self.protected_label)

                # a list of (weights, actual length)
                # Noticing that if stratified_g, the order within the key has been changed.
                condidate_selected = min([len(group_idx[k])/weighting_counter[k[0]] for k in group_idx.keys()])

                distinct_y_label = set(self.y)
                distinct_g_label = set(self.protected_label)
                
                # iterate each main task class
                # for y in distinct_y_label:
                for g in distinct_g_label:
                    selected = int(condidate_selected * weighting_counter[g])
                    # for g in distinct_g_label:
                    for y in distinct_y_label:
                        _index = group_idx[(g,y)]
                        shuffle(_index)
                        selected_index = selected_index + _index[:selected]


            X = [self.X[index] for index in selected_index]
            self.X = np.array(X)
            y = [self.y[index] for index in selected_index]
            self.y = np.array(y)
            gender_label = [self.protected_label[index] for index in selected_index]
            self.protected_label = np.array(gender_label)
        
        print("Done, loaded data shapes: {}, {}".format(self.X.shape, self.y.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        # return self.X[index], self.y[index], self.location_label[index], self.protected_label[index]
        
        if self.weighting is None:
            return self.X[index], self.y[index], self.protected_label[index]
        else:
            return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index]
    
    def load_data(self):
        data = pd.read_pickle(self.data_dir / self.filename)

        if self.protected_task in ["economy", "both"] and self.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]
 
        self.X = list(data[self.embedding_type])
        self.y = data["profession_class"].astype(np.float64) #Profession
        if self.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32) # Economy
        else:
            self.protected_label = data["intersection_class"].astype(np.int32) # Intersection

        return 0


if __name__ == "__main__":

    # split = "train"
    # test_data_set = biasbios(split)
    # # show data analysis
    # test_data_set.data_analysis()

    # for split in ["train", "dev", "test"]:
    #     for protected_attribute in ["gender", "economy", "both"]:
    #         print("#"*20)
    #         print("Split: {}".format(split))
    #         print("Protected label: {}".format(protected_attribute))
    #         print("#"*20)
    #         test_data_set = biasbios(split = split, protected_attribute = protected_attribute)
    #         test_data_set.data_analysis()

    # folder_dir = Path("/home/xudongh1/Project/joint_debiasing/data/bios")
    # folder_dir = Path(r"G:\Server_Back\doe_back\Project\joint_debiasing\data\bios")
    # folder_dir = Path(r"/data/cephfs/punim1421/Dataset/bios/bios_400k")
    # folder_dir = Path(r"D:\Datasets\biosbias-master")
    folder_dir = Path(r"D:\Project\Minding_Imbalance_in_Discriminator_Training\data\bios")
    
    split = "train"
    my_dataset = BiosDataset(folder_dir, split)
    
    # from collections import Counter
    # print(Counter(my_dataset.gender_label))
    # print(Counter(my_dataset.y))
    # print(len(Counter(my_dataset.y)))
    # print(max(np.array(list(Counter(my_dataset.gender_label)))))
    
    for weighting in ["ratio", "y", "g", "class", "class_g", "na"]:
        my_dataset = BiosDataset(folder_dir, split, weighting = weighting)
        print(weighting)
        print(len(my_dataset.y))
        print(my_dataset.instance_weights[:20])

    # for balance_type in ["y", "g", "ratio", "class", "stratified", "stratified_g"]:
    #     my_dataset = BiosDataset(
    #             data_dir = folder_dir, 
    #             split = "train", 
    #             embedding_type = "bert_avg_SE",
    #             weighting = None,
    #             balance_type = balance_type)
    #     print(len(my_dataset.y))
