
import torch
from torch import nn as nn
import os
import sys

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        input_size = 300 # Same as the hidden size of the main model
        num_classes = 2
        # args.dataset = "Moji"

        self.GR = False
        self.grad_rev = GradientReversal(args.adv_lambda)
        self.fc1 = nn.Linear(input_size, args.adv_units)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(args.adv_units, args.adv_units)
        self.fc3 = nn.Linear(args.adv_units, num_classes)

        self.init_training(args.dataset)
        self.adv_model_path = os.path.join(args.get_save_directory(), 'discriminator.pth')
        self.device = args.device

        torch.save(self.state_dict(), self.adv_model_path)

    def forward(self, input):
        if self.GR:
            input = self.grad_rev(input)
        out = self.fc1(input)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
    def Change_GradientReversal(self, State=True):
        self.GR = State
    
    def init_training(self, dataset):
        if dataset == "Moji":
            from fairness_src.dataloaders.deep_moji import DeepMojiDataset
            if sys.platform == "win32":
                Moji_data_path = "D:\\Project\\User_gender_removal\\data\\deepmoji\\split2\\"
            else:
                Moji_data_path = '/data/cephfs/punim1421/Dataset/deepmoji/split2/'

            train_data = DeepMojiDataset(Moji_data_path, "train", ratio=0.8, n = 10000)
            dev_data = DeepMojiDataset(Moji_data_path, "dev")
            test_data = DeepMojiDataset(Moji_data_path, "test")

        elif dataset == "Bios":
            from fairness_src.dataloaders.biography import BiosDataset
            from pathlib import Path
            if sys.platform == "win32":
                Bios_folder_dir = Path(r"D:\Project\Minding_Imbalance_in_Discriminator_Training\data\bios")
            else:
                Bios_folder_dir =Path(r'/data/cephfs/punim1421/Dataset/bios_gender_economy')

            train_data = BiosDataset(Bios_folder_dir, split = "train", weighting = None)
            dev_data = BiosDataset(Bios_folder_dir, split = "dev", weighting = None)
            test_data = BiosDataset(Bios_folder_dir, split = "test", weighting = None)
        
        tran_dataloader_params = {
            'batch_size': 4096,
            'shuffle': True,
            'num_workers': 0}

        eval_dataloader_params = {
            'batch_size': 4096,
            'shuffle': False,
            'num_workers': 0}

        self.training_generator = torch.utils.data.DataLoader(train_data, **tran_dataloader_params)
        self.validation_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
        self.test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

        from torch.optim import Adam
        self.adv_optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, model, model_params):
        self.GR = False
        # Train discriminator until converged
        # evaluate discriminator 
        best_adv_loss, _, _, _ = adv_eval_epoch(model, model_params, self, self.validation_generator, self.criterion, self.device)
        best_adv_epoch = -1
        for j in range(10):
            adv_train_epoch(model, model_params, self, self.training_generator, self.adv_optimizer, self.criterion, self.device)
            adv_valid_loss, _, _, _ = adv_eval_epoch(model, model_params, self, self.validation_generator, self.criterion, self.device)
            
            if adv_valid_loss < best_adv_loss:
                    best_adv_loss = adv_valid_loss
                    best_adv_epoch = j
                    torch.save(self.state_dict(), self.adv_model_path)
            else:
                if best_adv_epoch + 5 <= j:
                    break
        self.load_state_dict(torch.load(self.adv_model_path))

    def train_batch(self, model, model_params, batch_inputs, batch_protected_labels):
        self.adv_optimizer.zero_grad()
        hs = model.hidden_with_param(batch_inputs, model_params).detach()
        adv_predictions = self(hs)
        adv_loss = self.criterion(adv_predictions, batch_protected_labels)
        adv_loss.backward()
        self.adv_optimizer.step()
        self.adv_optimizer.zero_grad()

    def adv_loss(self, model, model_params, batch_inputs, batch_protected_labels):
        # hidden
        hs = model.hidden_with_param(batch_inputs, model_params)
        self.GR = True
        adv_predictions = self(hs)
        adv_loss = self.criterion(adv_predictions, batch_protected_labels)
        
        return adv_loss

class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# train a discriminator 1 epoch
def adv_train_epoch(model, model_parameters, discriminator, adv_iterator, adv_optimizer, criterion, device):
    """"
    Train the discriminator to get a meaningful gradient
    """    
    epoch_loss = 0
    epoch_acc = 0
    
    # model.train()
    # discriminator.train()

    # deactivate gradient reversal layer
    discriminator.GR = False
    
    for batch in adv_iterator:
        
        text = batch[0]
        tags = batch[1].long()
        p_tags = batch[2].long()

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        adv_optimizer.zero_grad()
        
        hs = model.hidden_with_param(text, model_parameters).detach()

        adv_predictions = discriminator(hs)

        loss = criterion(adv_predictions, p_tags)
                        
        loss.backward()
        
        adv_optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(adv_iterator)


# evaluate the discriminator
def adv_eval_epoch(model, model_parameters, discriminator, adv_iterator, criterion, device):
    """"
    Train the discriminator to get a meaningful gradient
    """

    epoch_loss = 0
    epoch_acc = 0
    
    # model.eval()
    # discriminator.eval()

    # deactivate gradient reversal layer
    discriminator.GR = False
    
    preds = []
    labels = []
    private_labels = []

    for batch in adv_iterator:
        
        text = batch[0]
        tags = batch[1].long()
        p_tags = batch[2].long()

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
                
        hs = model.hidden_with_param(text, model_parameters).detach()


        adv_predictions = discriminator(hs)

        # add the weighted loss

        loss = criterion(adv_predictions, p_tags)
                        
        loss.backward()

        epoch_loss += loss.item()
        
        adv_predictions = adv_predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(adv_predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
    
    return ((epoch_loss / len(adv_iterator)), preds, labels, private_labels)
