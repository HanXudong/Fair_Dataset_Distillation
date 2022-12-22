from torchvision import datasets, transforms
# from torchtext import datasets as textdata
# from torchtext import data
# from torchtext.vocab import GloVe
from PIL import Image
from .usps import USPS
from . import caltech_ucsd_birds
from . import pascal_voc
import os
import contextlib
import numpy as np
import torch
from collections import namedtuple
import math
import torch.nn as nn
from .utils import BaseDataset
from pathlib import Path
import sys

default_dataset_roots = dict(
    MNIST='./data/mnist',
    MNIST_RGB='./data/mnist',
    SVHN='./data/svhn',
    USPS='./data/usps',
    Cifar10='./data/cifar10',
    CUB200='./data/birds',
    PASCAL_VOC='./data/pascal_voc',
    imdb='./data/text/imdb',
    sst5='./data/text/sst',
    trec6='./data/text/trec',
    trec50='./data/text/trec',
    snli='./data/text/snli',
    multinli='./data/text/multinli',
    Moji='./data/fairness/Moji' if sys.platform == "win32" else '/data/cephfs/punim1421/Dataset/deepmoji/split2/',
    Bios='./data/fairness/Bios' if sys.platform == "win32" else '/data/cephfs/punim1421/Dataset/bios_gender_economy',
)


dataset_normalization = dict(
    MNIST=((0.1307,), (0.3081,)),
    MNIST_RGB=((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    USPS=((0.15972736477851868,), (0.25726667046546936,)),
    SVHN=((0.4379104971885681, 0.44398033618927, 0.4729299545288086),
          (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)),
    Cifar10=((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    CUB200=((0.47850531339645386, 0.4992702007293701, 0.4022205173969269),
            (0.23210887610912323, 0.2277066558599472, 0.26652416586875916)),
    PASCAL_VOC=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    imdb=((0,),(0,)),
    sst5=((0,),(0,)),
    trec6=((0,),(0,)),
    trec50=((0,),(0,)),
    snli=((0,),(0,)),
    multinli=((0,),(0,)),
    Moji=((0,),(0,)),
    Bios=((0,),(0,))
)


dataset_labels = dict(
    MNIST=list(range(10)),
    MNIST_RGB=list(range(10)),
    USPS=list(range(10)),
    SVHN=list(range(10)),
    Cifar10=('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'monkey', 'horse', 'ship', 'truck'),
    CUB200=caltech_ucsd_birds.class_labels,
    PASCAL_VOC=pascal_voc.object_categories,
    imdb={0,1},
    sst5=list(range(5)),
    trec6=list(range(6)),
    trec50=list(range(50)),
    snli=list(range(3)),
    multinli=list(range(3)),
    Moji=list(range(2)),
    Bios=list(range(28)),
)

# (nc, real_size, num_classes)
DatasetStats = namedtuple('DatasetStats', ' '.join(['nc', 'real_size', 'num_classes']))

dataset_stats = dict(
    MNIST=DatasetStats(1, 28, 10),
    MNIST_RGB=DatasetStats(3, 28, 10),
    USPS=DatasetStats(1, 28, 10),
    SVHN=DatasetStats(3, 32, 10),
    Cifar10=DatasetStats(3, 32, 10),
    CUB200=DatasetStats(3, 224, 200),
    PASCAL_VOC=DatasetStats(3, 224, 20),
    imdb = DatasetStats(1, 0, 2),
    sst5 = DatasetStats(1, 0, 5),
    trec6 = DatasetStats(1, 0, 6),
    trec50 = DatasetStats(1, 0, 50),
    snli = DatasetStats(1, 0, 3),
    multinli = DatasetStats(1, 0, 3),
    Moji = DatasetStats(1, 1, 2),
    Bios = DatasetStats(1, 1, 28),
)

assert(set(default_dataset_roots.keys()) == set(dataset_normalization.keys()) ==
       set(dataset_labels.keys()) == set(dataset_stats.keys()))

def print_closest_words(vec, glove, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[0:n+1]: 					       # take the top n
        print(glove.itos[idx], difference)
def closest_words(vec, glove, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    return [glove.itos[idx] for idx, _ in lst[0:n+1]]				       # take the top n
def get_info(state):
    dataset_stats['imdb']=DatasetStats(1,state.maxlen,2)
    dataset_stats['sst5']=DatasetStats(1,state.maxlen,5)
    dataset_stats['trec6']=DatasetStats(1,state.maxlen,6)
    dataset_stats['trec50']=DatasetStats(1,state.maxlen,50)
    dataset_stats['snli']=DatasetStats(1,state.maxlen,3)
    dataset_stats['multinli']=DatasetStats(1,state.maxlen,3)
    name = state.dataset  # argparse dataset fmt ensures that this is lowercase and doesn't contrain hyphen
    assert name in dataset_stats, 'Unsupported dataset: {}'.format(state.dataset)
    nc, input_size, num_classes = dataset_stats[name]
    normalization = dataset_normalization[name]
    root = state.dataset_root
    if root is None:
        root = default_dataset_roots[name]
    labels = dataset_labels[name]
    return name, root, nc, input_size, num_classes, normalization, labels


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


def get_dataset(state, phase):
    dataset_stats['imdb']=DatasetStats(1,state.maxlen,2)
    dataset_stats['sst5']=DatasetStats(1,state.maxlen,5)
    dataset_stats['trec6']=DatasetStats(1,state.maxlen,6)
    dataset_stats['trec50']=DatasetStats(1,state.maxlen,50)
    dataset_stats['snli']=DatasetStats(1,state.maxlen,3)
    dataset_stats['multinli']=DatasetStats(1,state.maxlen,3)
    assert phase in ('train', 'test'), 'Unsupported phase: %s' % phase
    name, root, nc, input_size, num_classes, normalization, _ = get_info(state)
    real_size = dataset_stats[name].real_size

    state.data_dir = root

    if name == 'MNIST':
        if input_size != real_size:
            transform_list = [transforms.Resize([input_size, input_size], Image.BICUBIC)]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == 'train'), download=True,
                                  transform=transforms.Compose(transform_list))
    elif name == 'MNIST_RGB':
        transform_list = [transforms.Grayscale(3)]
        if input_size != real_size:
            transform_list.append(transforms.Resize([input_size, input_size], Image.BICUBIC))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == 'train'), download=True,
                                  transform=transforms.Compose(transform_list))
    elif name == 'USPS':
        if input_size != real_size:
            transform_list = [transforms.Resize([input_size, input_size], Image.BICUBIC)]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return USPS(root, train=(phase == 'train'), download=True,
                        transform=transforms.Compose(transform_list))
    elif name == 'SVHN':
        transform_list = []
        if input_size != real_size:
            transform_list.append(transforms.Resize([input_size, input_size], Image.BICUBIC))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.SVHN(root, split=phase, download=True,
                                 transform=transforms.Compose(transform_list))
    elif name == 'Cifar10':
        transform_list = []
        if input_size != real_size:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        if phase == 'train':
            transform_list += [
                # TODO: merge the following into the padding options of
                #       RandomCrop when a new torchvision version is released.
                transforms.Pad(padding=4, padding_mode='reflect'),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.CIFAR10(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'CUB200':
        transform_list = []
        if phase == 'train':
            transform_list += [
                transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        return caltech_ucsd_birds.CUB200(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'PASCAL_VOC':
        transform_list = []
        if phase == 'train':
            transform_list += [
                transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        if phase == 'train':
            phase = 'trainval'
        return pascal_voc.PASCALVoc2007(root, phase, transforms.Compose(transform_list))
    elif name == 'imdb':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, test = textdata.IMDB.splits(TEXT, LABEL)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #man=TEXT.vocab.vectors[TEXT.vocab["man"]].clone()
        #woman=TEXT.vocab.vectors[TEXT.vocab["woman"]].clone()
        #king=TEXT.vocab.vectors[TEXT.vocab["doctor"]].clone()
        
        #print(torch.norm(king - man + woman))
        #vec = king - man + woman
        #print_closest_words(vec, TEXT.vocab)
        #print_closest_words(king, TEXT.vocab)
        #print(TEXT.vocab.vectors)
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'sst5':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, valid, test = textdata.SST.splits(TEXT, LABEL, fine_grained=True)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'trec6':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, test = textdata.TREC.splits(TEXT, LABEL, fine_grained=False)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'trec50':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, test = textdata.TREC.splits(TEXT, LABEL, fine_grained=True)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'snli':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, valid, test = textdata.SNLI.splits(TEXT, LABEL)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'multinli':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, valid, test = textdata.MultiNLI.splits(TEXT, LABEL)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'Moji':
        # class DeepMojiDataset(torch.utils.data.Dataset):
        #     def __init__(self, split, root) -> None:

        #         # TODO: add to arguments.
        #         # Hard code hyperparameters
        #         self.data_dir = Path(root) / split
        #         self.ratio = 0.8
        #         self.n = 100000
        #         self.p_aae = 0.5

        #                 # Init 
        #         self.X = []
        #         self.y = []
        #         self.private_label = []

        #         # Load preprocessed tweets, labels, and tweet ids.
        #         print("Loading preprocessed deepMoji Encoded data")
        #         self.load_data()

        #         self.X = np.array(self.X)
        #         self.y = np.array(self.y)
        #         self.private_label = np.array(self.private_label)

        #     def __len__(self):
        #         'Denotes the total number of samples'
        #         return len(self.y)

        #     def __getitem__(self, index):
        #         'Generates one sample of data'

        #         return self.X[index], self.y[index], self.private_label[index]

        #     def load_data(self):
        #         n_1 = int(self.n * self.ratio / 2)
        #         n_2 = int(self.n * (1 - self.ratio) / 2)

        #         for file, label, private, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
        #                                                 [1, 1, 0, 0],
        #                                                 [1, 0, 1, 0],
        #                                                 [n_1, n_2, n_2, n_1]):
        #             data = np.load(self.data_dir / '{}.npy'.format(file))
        #             # print(data.shape)
        #             data = list(data[:class_n])
        #             self.X = self.X + data
        #             self.y = self.y + [label]*len(data)
        #             self.private_label = self.private_label + [private]*len(data)
        # return DeepMojiDataset(phase, root)

        class DeepMojiDataset(BaseDataset):

            p_aae = 0.5 # distribution of the main label, proportion of the AAE
            n = 100000 # target size

            def load_data(self):
                # stereotyping, 0.5 is balanced 
                if self.split == "train":
                    self.ratio = 0.8 
                else:
                    self.ratio = 0.5 # stereotyping, 0.5 is balanced 

                self.data_dir = Path(self.args.data_dir) / self.split

                n_1 = int(self.n * self.p_aae * self.ratio) # happy AAE 
                n_2 = int(self.n * (1-self.p_aae) * (1-self.ratio)) # happy SAE
                n_3 = int(self.n * self.p_aae * (1-self.ratio)) # unhappy AAE
                n_4 = int(self.n * (1-self.p_aae) * self.ratio) # unhappy SAE


                for file, label, protected, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                                            [1, 1, 0, 0],
                                                                            [1, 0, 1, 0], 
                                                                            [n_1, n_2, n_3, n_4]
                                                                            ):
                    data = np.load('{}/{}.npy'.format(self.data_dir, file))
                    data = list(data[:class_n])
                    self.X = self.X + data
                    self.y = self.y + [label]*len(data)
                    self.protected_label = self.protected_label + [protected]*len(data)
        
        return DeepMojiDataset(state, phase)


    elif name == 'Bios':
        # Init the file name
        # filename = "{}.pkl".format(phase)

        # class BiosDataset(torch.utils.data.Dataset):
        #     def __init__(self, filename, root) -> None:

        #         # TODO: add to arguments.
        #         # Hard code hyperparameters
        #         embedding_type = "bert_avg_SE"

        #         import pandas as pd
        #         from pathlib import Path

        #         _data = pd.read_pickle(Path(root) / filename)

        #         X = list(_data[embedding_type])
        #         y = _data["profession_class"].astype(np.float64)
        #         protected_label = _data["gender_class"].astype(np.int32)

        #         self.X = np.array(X)
        #         if len(self.X.shape) == 3:
        #             self.X = np.concatenate(list(self.X), axis=0)
        #         self.y = np.array(y)
        #         self.protected_label = np.array(protected_label)

        #     def __len__(self):
        #         'Denotes the total number of samples'
        #         return len(self.y)

        #     def __getitem__(self, index):
        #         'Generates one sample of data'

        #         return self.X[index], self.y[index], self.protected_label[index]

        class BiosDataset(BaseDataset):
            embedding_type = "bert_avg_SE"
            text_type = "hard_text"

            def load_data(self):
                import pandas as pd

                self.filename = "bios_{}_df.pkl".format(self.split)

                data = pd.read_pickle(Path(self.args.data_dir) / self.filename)
                
                # hard code the protected task as gender
                self.protected_task = "gender"

                if self.protected_task in ["economy", "both"] and self.args.full_label:
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
        
        return BiosDataset(state, phase)

    else:
        raise ValueError('Unsupported dataset: %s' % state.dataset)
