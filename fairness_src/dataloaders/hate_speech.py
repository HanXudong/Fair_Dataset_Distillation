import logging
from typing import Dict

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from collections import defaultdict, Counter
from random import shuffle


def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, 
                args, 
                data_dir, 
                split, 
                full_label_instances = False,
                protected_label = "age",
                embedding_type = "text_hidden",
                weighting = None,
                balance_type = None,
                upsampling = False,
                size = None,
                subsampling = False,
                subsampling_ratio = 1
                ):
        self.args = args
        self.data_dir = data_dir
        self.dataset_type = {"train", "valid", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, valid, and test"
        self.split = split

        # check protected attribute
        assert protected_label in ["gender", "age", "country", "ethnicity", "intersection"]        
        self.embedding_type = embedding_type

        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed Encoded data")
        if not full_label_instances:
            df, text_embedding, label, author_protected_label_columns, total_n, selected_n= self.load_dataset(
                not_nan_col = [],
                author_protected_labels = [protected_label]
                )
        else:
            if protected_label != "intersection":
                df, text_embedding, label, author_protected_label_columns, total_n, selected_n = self.load_dataset(
                    not_nan_col = ["gender", "age", "country", "ethnicity"],
                    author_protected_labels = [protected_label]
                    )
            else:    
                df, text_embedding, label, author_protected_label_columns, total_n, selected_n = self.load_dataset(
                    not_nan_col = [protected_label],
                    author_protected_labels = [protected_label]
                    )

        self.X = text_embedding
        self.y = label
        self.protected_label = author_protected_label_columns[protected_label]

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.protected_label = np.array(self.protected_label)

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

        # # Upsampling
        # if full_label_instances and upsampling:
        #     if size == None:
        #         # Keep using the same size by default
        #         pass
        #     else:
        #         # Use the target size
        #         total_n = size
            
        #     # upsampling with replacement to get a same size dataset
        #     selected_indices = np.random.choice([i for i in range(selected_n)], replace=True, size=total_n)

        # elif full_label_instances and subsampling:
        #     # keep the same total number
        #     # reduce the number of distinct instances with private labels
        #     # 0 <= subsampling_ratio <= 1
        #     sub_indices = np.random.choice([i for i in range(selected_n)], replace=False, size = int(subsampling_ratio*selected_n))
        #     print("Number of distinct instances: {}".format(len(sub_indices)))
            
        #     selected_indices = np.random.choice(sub_indices, replace=True, size=selected_n)
            
        # else:
        #     selected_indices = np.array([i for i in range(len(self.y))])

        # # update values
        # self.X = self.X[selected_indices]
        # self.y = self.y[selected_indices]
        # self.protected_label = self.protected_label[selected_indices]

        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.weighting is None:
            return self.X[index], self.y[index], self.protected_label[index]
        else:
            return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index]

    def load_dataset(
        self,
        not_nan_col = [],
        author_protected_labels = []
        ):
        dataset_dir = self.data_dir / "hateSpeech_{}_df.pkl".format(self.split)
        df = pd.read_pickle(dataset_dir)
        # df = df.replace("x", np.nan)

        total_n = len(df)

        df = df[full_label_data(df, not_nan_col)]

        selected_n = len(df)

        print("Select {} out of {} in total.".format(selected_n, total_n))

        # input and labels
        text_embedding = list(df[self.embedding_type])
        label = list(df["label"])
        # protected_labels
        author_protected_label_columns = {
            p_name:list(df[p_name].astype("float"))
            for p_name in author_protected_labels
        }

        for p_name in author_protected_labels:
            df[p_name] = author_protected_label_columns[p_name]

        return df, text_embedding, label, author_protected_label_columns, total_n, selected_n


if __name__ == "__main__":
    class Args:
        gender_balanced = False
    
    # data_path = "D:\\Project\\User_gender_removal\\data\\deepmoji\\"
    # data_path = "D:\\Project\\adv_decorrelation\\data\\hate_speech"
    data_path = Path(r"D:\Project\Minding_Imbalance_in_Discriminator_Training\data\hateSpeech")

    for p_label in ["gender", "age", "ethnicity", "country", "intersection"]:
        for full_label_flag in [True]:
            for balance_type in ["y", "g", "ratio", "class", "stratified", "stratified_g"]:
                print(full_label_flag, p_label, balance_type)
                split = "train"
                args = Args()
                my_dataset = HateSpeechDataset(\
                    args, 
                    data_path, 
                    split, 
                    full_label_instances = full_label_flag, 
                    protected_label = p_label,
                    embedding_type = "text_hidden",
                    weighting = None,
                    balance_type = balance_type)
                print(len(my_dataset.y))