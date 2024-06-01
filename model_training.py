# This file focuses on training the FC (2 layers) model with the full training set. The model is named BioNN in our
# code framework.

# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizerFast, BertModel

# load python files
from extract_features import load_features, load_features_by_name
from models import BioNN, BioDeepNN, BioResNet

# define constants
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# define dataset and dataloader for Neural Network Model (without BERT)
class PeptidesDataset(Dataset):
    # Dataset for peptides, features and labels
    def __init__(self, peptides, features, labels):
        self.peptides = peptides
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        return peptide, feature, label


class PeptidesDataLoader(DataLoader):
    # Dataloader for Neural Network (without BERT)
    def __init__(self, peptides, features, labels, batch_size, shuffle=True):
        dataset = PeptidesDataset(peptides, features, labels)
        super().__init__(dataset, batch_size, shuffle)


class BertDataset(Dataset):
    # This dataset is used to store the data for training DL models. If you want to use it to load data, please use
    # the dataloader following this class.
    def __init__(self, input_ids, attention_mask, token_type_ids, features, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_m = self.attention_mask[idx]
        token_type_id = self.token_type_ids[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        return input_id, attention_m, token_type_id, feature, label


class BertDataLoader(DataLoader):
    # This dataloader is used to load data for training DL models.
    # The usage example can be found in dl_experiment.py and model_training.py
    def __init__(self, input_ids, attention_mask, token_type_ids, features, labels, batch_size, shuffle=True):
        dataset = BertDataset(input_ids, attention_mask, token_type_ids, features, labels)
        super().__init__(dataset, batch_size, shuffle)


if __name__ == "__main__":
    # import data and preprocess
    data = pd.read_csv("./Data/processed_peptides10.csv")  # load data

    # get peptide sequences
    peptides = data.iloc[:, 0].values.tolist()  # ['PEPTIDE1', 'PEPTIDE2', ...]

    # load extracted features
    features_x = load_features().iloc[:, 1:].values
    features_x = torch.from_numpy(features_x).float()  # convert features_x to tensor

    # import Bert-Base-Proteins model
    checkpoint = 'unikei/bert-base-proteins'  # load the checkpoint of bert
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)  # load tokenizer
    bert_model = BertModel.from_pretrained(checkpoint).to(device)  # load bert model

    # process mmp labels
    labels = data.iloc[:, 1:].values
    labels = torch.from_numpy(labels).float().to(device)  # convert labels to tensor

    # tokenize the peptide sequences
    tokens = tokenizer(peptides, return_tensors='pt')
    input_ids, attention_mask, token_type_ids = tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]

    # build training dataloader
    train_dataloader = BertDataLoader(input_ids, attention_mask, token_type_ids, features_x, labels, 512, shuffle=True)

    # initialize the FC (2 layers) models
    bio_model = BioNN(768 * 10 + features_x.shape[1], labels_num=18).to(
        device)  # the input size is 768 * 10 + features_x.shape[1]

    # train the model
    # setup optimizer
    train_bert_params = [id(bert_model.pooler.dense.bias), id(bert_model.pooler.dense.weight)]
    bert_params = filter(lambda p: id(p) not in train_bert_params,
                         bert_model.parameters())  # separate newly initialized params from all bert params
    optimizer = optim.Adam(
        [
            {"params": bert_params, "lr": 1e-6},  # finetune the bert model with learning rate 1e-6
            {"params": bert_model.pooler.dense.bias, "lr": 1e-4},  # train the model with learning rate 1e-4
            {"params": bert_model.pooler.dense.weight, "lr": 1e-4},
            {"params": bio_model.parameters(), "lr": 1e-4}
        ], lr=1e-4
    )

    # setup loss function
    criterion = nn.MSELoss()

    # set train mode
    bert_model.train()
    bio_model.train()

    # start to train
    train_epochs = 100
    pbar = tqdm(range(train_epochs), desc="Training: ", total=train_epochs)
    pbar.set_postfix(loss=0)
    loss_track = []
    for epoch in pbar:
        loss_track_epoch = []
        for epoch_input_ids, epoch_attention_mask, epoch_token_type_ids, features_epoch, labels_epoch in train_dataloader:
            optimizer.zero_grad()

            # set input to GPU
            epoch_input_ids = epoch_input_ids.to(device)
            epoch_attention_mask = epoch_attention_mask.to(device)
            epoch_token_type_ids = epoch_token_type_ids.to(device)

            bert_output = bert_model(input_ids=epoch_input_ids, attention_mask=epoch_attention_mask,
                                     token_type_ids=epoch_token_type_ids).last_hidden_state.view(
                features_epoch.shape[0], -1)  # flatten the embedding result

            bio_input = torch.cat([bert_output, features_epoch.to(device)], dim=1)  # concat BERT output with feature
            bio_output = bio_model(bio_input)

            loss = criterion(bio_output.view(labels_epoch.size()), labels_epoch)  # calculate loss
            loss.backward()

            optimizer.step()

            loss_track_epoch.append(loss.detach().to("cpu").item())  # track the loss

        avg_loss = np.average(loss_track_epoch)
        loss_track.append(avg_loss)
        pbar.set_postfix(loss=avg_loss)

    loss_track = np.array(loss_track)

    # save the models
    torch.save(bert_model.state_dict(), "./Model/bert_model.pth")
    torch.save(bio_model.state_dict(), "./Model/bio_model.pth")
    np.save("./Model/loss_track.npy", loss_track)  # save the losses during training
