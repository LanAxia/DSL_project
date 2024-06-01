# Validate the Deep Learning Models by KFold Cross-Validation

# import packages
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizerFast, BertModel

# import other files
from extract_features import load_features
from models import BioNN, BioDeepNN, BioResNet
from model_training import BertDataLoader


def validate_dl(model_class: nn.Module, peptides: list, features: np.array, y: np.array) -> np.array:
    # This function is used to validate the model effect by KFold cross-validation. (k=5)
    # return a (5, 100, 2, data_size) np.array
    # dim 0: 5 folds
    # dim 1: 100 epochs
    # dim 2: 0 is pred, 1 is truth
    # dim 3: data size
    validation = np.zeros((100, 2, y.shape[0], 18))

    # build KFold
    kf = KFold(n_splits=5, random_state=33, shuffle=True)
    tokens = tokenizer(peptides, return_tensors='pt')  # tokenize peptides
    input_ids, attention_mask, token_type_ids = tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]
    for i, (train_id, test_id) in enumerate(kf.split(features)):
        # split training data and validation data
        train_input_ids, train_attention_mask, train_token_type_ids, train_features, train_y = input_ids[train_id], \
            attention_mask[train_id], token_type_ids[train_id], torch.from_numpy(
            features[train_id]).float(), torch.from_numpy(y[train_id]).float()
        test_input_ids, test_attention_mask, test_token_type_ids, test_features, test_y = input_ids[test_id], \
            attention_mask[test_id], token_type_ids[test_id], torch.from_numpy(
            features[test_id]).float(), torch.from_numpy(
            y[test_id]).float()

        # build training dataloader and validation dataloader
        train_dataloader = BertDataLoader(train_input_ids, train_attention_mask, train_token_type_ids, train_features,
                                          train_y, 512, shuffle=True)
        test_dataloader = BertDataLoader(test_input_ids, test_attention_mask, test_token_type_ids, test_features,
                                         test_y, 512, shuffle=False)

        # initialize the model
        bert_model = BertModel.from_pretrained(checkpoint).to(device)
        bio_model = model_class(768 * 10 + features.shape[1]).to(device)

        # set optimizer
        train_bert_params = [id(bert_model.pooler.dense.bias), id(bert_model.pooler.dense.weight)]
        bert_params = filter(lambda p: id(p) not in train_bert_params, bert_model.parameters())
        optimizer = optim.Adam(
            [
                {"params": bert_params, "lr": 1e-6},
                {"params": bert_model.pooler.dense.bias, "lr": 1e-4},
                {"params": bert_model.pooler.dense.weight, "lr": 1e-4},
                {"params": bio_model.parameters(), "lr": 1e-4},
            ], lr=1e-4
        )

        # set criterion
        criterion = nn.MSELoss()

        # start to train
        # set train mode
        bert_model.train()
        bio_model.train()

        # set training epochs
        train_epochs = 100
        for epoch in tqdm(range(train_epochs), total=train_epochs):
            epoch_pointer = 0
            for batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()

                # set input to GPU
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_token_type_ids = batch_token_type_ids.to(device)

                bert_output = bert_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask,
                                         token_type_ids=batch_token_type_ids).last_hidden_state.view(
                    batch_features.shape[0], -1)  # flatten the BERT output
                bio_input = torch.cat([bert_output, batch_features.to(device)], dim=1)  # concat with features

                bio_output = bio_model(bio_input)

                loss = criterion(bio_output.view(batch_labels.size()),
                                 batch_labels.to(device))  # calculate loss
                loss.backward()

                optimizer.step()

            # validate after each epoch
            bert_model.eval()  # set evaluation mode
            bio_model.eval()
            pred = []
            with torch.no_grad():  # inference without gradient
                for batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_features, batch_labels in test_dataloader:
                    # set input to GPU
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_token_type_ids = batch_token_type_ids.to(device)

                    bert_output = bert_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask,
                                             token_type_ids=batch_token_type_ids).last_hidden_state.view(
                        batch_features.shape[0], -1)  # flatten the BERT output
                    bio_input = torch.cat([bert_output, batch_features.to(device).float()],
                                          dim=1)  # concat with features

                    bio_output = bio_model(bio_input)
                    pred.append(bio_output.to("cpu"))

            # save the validation result to validation
            epoch_pred = torch.cat(pred, dim=0).detach().numpy()  # concat the prediction of multiple batches
            epoch_true = test_y.to("cpu").numpy()
            validation[epoch, 0, test_id, :] = epoch_pred
            validation[epoch, 1, test_id, :] = epoch_true
            epoch_pointer += epoch_pred.shape[0]

            bert_model.train()
            bio_model.train()

        print("Fold {} finished".format(i))
    return validation


if __name__ == "__main__":
    # constant
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # process data
    # load data
    data = pd.read_csv("./Data/processed_peptides10.csv")

    # get peptide sequences
    peptides = data.iloc[:, 0].values.tolist()  # ['Peptide1', 'Peptide2', ...]

    # load extracted features
    features_x = load_features().iloc[:, 1:].values

    # extract labels
    all_mmp_y = data.iloc[:, 1:].values

    # import Bert-Base-Protein model
    checkpoint = 'unikei/bert-base-proteins'
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

    # validate FC (2 layers) model
    bionn_validation = validate_dl(BioNN, peptides, features_x, all_mmp_y)
    np.save("./Result/BioNN_validation.npy", bionn_validation)

    # validate FC (4 layers) model
    biodnn_validation = validate_dl(BioDeepNN, peptides, features_x, all_mmp_y)
    np.save("./Result/BioDNN_validation.npy", biodnn_validation)

    # validate ResNet model
    biores_validation = validate_dl(BioResNet, peptides, features_x, all_mmp_y)
    np.save("./Result/BioRes_validation.npy", biores_validation)
