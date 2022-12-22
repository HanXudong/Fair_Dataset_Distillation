import torch.nn as nn
import torch.nn.functional as F
import logging
import itertools
from . import utils
import math
import torch

class MLPClassifier(utils.ReparamModule):
    # Support dim, will be used for model initialization
    supported_dims = set(range(1,2000000))

    def __init__(self, state):
        super(MLPClassifier, self).__init__()
        
        self.emb_size = state.ninp
        self.hidden_size = 300
        self.dropout = nn.Dropout(p=0)

        self.AF = nn.Tanh()

        self.dense1 = nn.Linear(self.emb_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, input):
        out = self.dense1(input)
        out = self.AF(out)
        out = self.dropout(out)
        out = self.dense2(out)
        out = self.AF(out)
        out = self.dense3(out)
        return out
    
    def hidden(self, input):
        out = self.dense1(input)
        out = self.AF(out)
        out = self.dropout(out)
        out = self.dense2(out)
        # out = self.AF(out)
        # out = self.dense3(out)
        return out


class LeNet(utils.ReparamModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        if state.dropout:
            raise ValueError("LeNet doesn't support dropout")
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class TextRNN1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextRNN1, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 100
        n_layers = 2
        bidirectional=True
        dropout=0.5 if self.state.mode=="train" else 0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding

        self.rnn = nn.RNN(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           bias =True,
                           batch_first=True,
                           nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        if self.state.mode=="train":
            out = self.dropout(out)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(out)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)
class TextLSTM2(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextLSTM2, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 100
        n_layers = 2
        bidirectional=True
        dropout=0.5 if self.state.mode=="train" else 0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           bias =True,
                           batch_first=True,
                           )
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        if self.state.mode=="train":
            out = self.dropout(out)
        self.rnn.flatten_parameters()
        out, (hidden,cell) = self.rnn(out)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

class TextRNN2(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextRNN2, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 100
        n_layers = 1
        dropout=0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = False

        self.rnn = nn.RNN(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=False,
                           dropout=dropout,
                           bias =True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        #out = self.dropout(out)
        out, hidden = self.rnn(out)
        #assert torch.equal(out[:,-1,:], hidden.squeeze(0))

        return self.sigm(self.fc(hidden.squeeze(0)))
class TextLSTM1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextLSTM1, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 10
        n_layers = 1
        dropout=0.7 if self.state.mode=="train" else 0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = True

        self.rnn = nn.LSTM(int(state.ninp/4)-1,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=False,
                           dropout=dropout,
                           bias =True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False
        self.conv1 = nn.Conv1d(state.maxlen, 16, 5)
        self.relu=nn.ReLU()
        self.maxpool = nn.MaxPool1d(4)

    def forward(self, x):
        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        if self.state.mode=="train":
            out = self.dropout(out)
        out = self.relu(self.conv1(out))
        out = self.maxpool(out)
        self.rnn.flatten_parameters()
        out, (hidden, cell) = self.rnn(out)
        #print (out.size())
        #print (hidden.size())
        #assert torch.equal(out[:,-1,:], hidden[-1,:,:].squeeze(0))
        return self.fc(hidden[-1,:,:].squeeze(0))
        #return self.sigm(self.fc(hidden.squeeze(0)))

class Transformer1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(Transformer1, self).__init__()
        self.output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        nhead=4
        hidden_dim = embedding_dim
        n_layers = 4
        dropout=0.1
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding
        self.decoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)
        self.classifier_head = nn.Linear(hidden_dim, self.output_dim)
        #self.sigm=nn.Sigmoid()
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        tgt_size=[i for i in out.size()]
        tgt_size[-2]=1
        #print(tgt_size)
        tgt=torch.rand(tgt_size).to(self.state.device)
        hidden = self.decoder(tgt, out).squeeze(1)
        return self.classifier_head(hidden)
       
class TextConvNet1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextConvNet1, self).__init__()
        #if state.textdata:
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        n_filters = 100
        filter_sizes = [3,4,5]
        dropout=0.5
        output_dim=1 if state.num_classes == 2 else state.num_classes
        self.encoder = nn.Embedding(ntoken, embedding_dim)
        self.encoder.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.encoder.weight.requires_grad = state.learnable_embedding 
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.sigm=nn.Sigmoid()
        self.distilling_flag=False
    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                out = self.encoder(x) #* math.sqrt(ninp)
                out.unsqueeze_(1)
                #out=x
                #print(out.size())
                #print(out.size())
        else:
                out=x
        #out = self.dropout(out)
        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        return self.fc(cat)     
    
class AlexCifarNet(utils.ReparamModule):
    supported_dims = {32}

    def __init__(self, state):
        super(AlexCifarNet, self).__init__()
        assert state.nc == 3
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# ImageNet
class AlexNet(utils.ReparamModule):
    supported_dims = {224}

    class Idt(nn.Module):
        def forward(self, x):
            return x

    def __init__(self, state):
        super(AlexNet, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if state.dropout:
            filler = nn.Dropout
        else:
            filler = AlexNet.Idt
        self.classifier = nn.Sequential(
            filler(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            filler(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if state.num_classes <= 2 else state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    

