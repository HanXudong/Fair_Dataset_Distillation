import logging
from typing import Dict

import numpy as np

import torch
import torch.utils.data as data

class DeepMojiDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, split, p_aae: float =  0.5, 
        ratio: float = 0.5, n: int = 100000, weighting = None, 
        adv_weighting = None,
        reverse_label = False, resampling: int = 0
        ):
        # self.args = args
        self.data_dir = data_dir
        self.dataset_type = {"train", "dev", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, dev, and test"
        self.split = split
        self.data_dir = self.data_dir+self.split
        self.p_aae = p_aae # distribution of the main label, proportion of the AAE
        self.ratio = ratio # stereotyping, 0.5 is balanced 
        self.n = n # target size
        self.weighting = weighting
        self.adv_weighting = adv_weighting
        # Init 
        self.reverse_label = reverse_label
        self.X = []
        self.y = []
        self.private_label = []
        self.instance_weights = []
        self.adv_instance_weights = []

        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed deepMoji Encoded data")
        self.load_data()

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)
        self.instance_weights = np.array(self.instance_weights)
        self.adv_instance_weights = np.array(self.adv_instance_weights)

        # Resampling to get a same size training set
        if resampling == 0:
            pass # don't need to resample the dataset
        else:
            # Resampling with replacement to get a dataset with the same size to the full dataset.
            number_indices = len(self.y)
            
            # upsampling with replacement to get a same size dataset
            sub_indices = np.random.choice([i for i in range(number_indices)], replace=True, size = int(resampling))
            selected_indices = np.random.choice(sub_indices, replace=True, size=resampling)
            # update values
            self.X = self.X[selected_indices]
            self.y = self.y[selected_indices]
            self.private_label = self.private_label[selected_indices]
            self.instance_weights = self.instance_weights[selected_indices]
            self.adv_instance_weights = self.adv_instance_weights[selected_indices]


        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.private_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.weighting is None:
            return self.X[index], self.y[index], self.private_label[index]
        else:
            return self.X[index], self.y[index], self.private_label[index], self.instance_weights[index], self.adv_instance_weights[index]
    
    def load_data(self):
        """
        Based on https://github.com/shauli-ravfogel/nullspace_projection/blob/2403d09a5d84b3ae65129b239331a22f89ad69fc/src/deepmoji/deepmoji_debias.py#L24
        """
        # ratios for pos / neg
        # n_1 = int(self.n * self.ratio / 2)
        # n_2 = int(self.n * (1 - self.ratio) / 2)
        n_1 = int(self.n * self.p_aae * self.ratio) # happy AAE 
        n_2 = int(self.n * (1-self.p_aae) * (1-self.ratio)) # happy SAE
        n_3 = int(self.n * self.p_aae * (1-self.ratio)) # unhappy AAE
        n_4 = int(self.n * (1-self.p_aae) * self.ratio) # unhappy SAE

        if self.weighting is not None:
            perfect_balanced_n = int(self.n / 4) 
            weights = perfect_balanced_n / np.array([n_1, n_2, n_3, n_4])
        else:
            weights = [1,1,1,1]
        
        if self.adv_weighting is not None:
            perfect_balanced_n = int(self.n / 4)
            adv_weights = perfect_balanced_n / np.array([n_1, n_2, n_3, n_4])
        else:
            adv_weights = [1,1,1,1]

        for file, label, private, class_n, weight, adv_weight in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                                    [1, 1, 0, 0] if not self.reverse_label else [0, 0, 1, 1], # reverse main task label 
                                                                    [1, 0, 1, 0], 
                                                                    [n_1, n_2, n_3, n_4],
                                                                    weights,
                                                                    adv_weights
                                                                    ):
            data = np.load('{}/{}.npy'.format(self.data_dir, file))
            # print(data.shape)
            data = list(data[:class_n])
            self.X = self.X + data
            self.y = self.y + [label]*len(data)
            self.private_label = self.private_label + [private]*len(data)
            self.instance_weights = self.instance_weights + [weight]*len(data)
            self.adv_instance_weights = self.adv_instance_weights + [adv_weight]*len(data)
        
        # Print dataset distribution
        # print(n_1, n_2, n_3, n_4)
        # print("p(happy|AAE): {}".format(n_1/(n_1+n_3)))
        # print("p(unhappy|SAE): {}".format(n_4/(n_2+n_4)))
        # print("p(AAE): {}".format((n_1+n_3)/(n_1+n_3+n_2+n_4)))
        # print("p(Happy): {}".format((n_1+n_2)/(n_1+n_3+n_2+n_4)))

if __name__ == "__main__":
 
    
    data_path = "D:\\Project\\User_gender_removal\\data\\deepmoji\\split2\\"
    split = "train"
    _ = DeepMojiDataset(data_path, split, p_aae =  0.2, ratio = 0.8, n = 50000)