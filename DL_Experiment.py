# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizerFast, BertModel


# 导入其他文件
from extract_features import load_features
from models import BioNN, BioDeepNN, BioResNet
from Model_Training import PeptidesDataLoader, BertDataLoader

# constant
SAVE = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 处理数据
# import data and preprocess
data = pd.read_csv("./Data/processed_peptides10.csv")  # load data

# 得到氨基酸序列
peptides = data.iloc[:, 0].values.tolist()  # 肽链的列表（字符串）

# load extracted features
features_x = load_features().iloc[:, 1:].values

# 处理mmp y的数据
all_mmp_y = data.iloc[:, 1:].values

# import Bert-Base-Protein model
checkpoint = 'unikei/bert-base-proteins'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

# 定义验证模型效果的函数
def validate_dl(model_class: nn.Module, peptides: list, features: np.array, y: np.array) -> np.array:
    # 返回一个(5, 100, 2, data_size)的np.array，其中2维中0是pred，1是true
    validation = np.zeros((100, 2, y.shape[0], 18))
    # 用5折交叉验证来验证模型效果
    kf = KFold(n_splits=5, random_state=33, shuffle=True)
    rg_errors = np.zeros((5, y.shape[1], 3))
    cl_errors = np.zeros((5, y.shape[1], 4))  # 评价指标包括auc, f1, precision, recall
    tokens = tokenizer(peptides, return_tensors='pt')
    input_ids, attention_mask, token_type_ids = tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]
    for i, (train_id, test_id) in enumerate(kf.split(features)):
        train_input_ids, train_attention_mask, train_token_type_ids, train_features, train_y = input_ids[train_id], attention_mask[train_id], token_type_ids[train_id], torch.from_numpy(features[train_id]).float(), torch.from_numpy(y[train_id]).float()
        test_input_ids, test_attention_mask, test_token_type_ids, test_features, test_y = input_ids[test_id], attention_mask[test_id], token_type_ids[test_id], torch.from_numpy(features[test_id]).float(), torch.from_numpy(y[test_id]).float()

        train_dataloader = BertDataLoader(train_input_ids, train_attention_mask, train_token_type_ids, train_features, train_y, 512, shuffle=True)
        test_dataloader = BertDataLoader(test_input_ids, test_attention_mask, test_token_type_ids, test_features, test_y, 512, shuffle=False)

        # 初始化模型
        bert_model = BertModel.from_pretrained(checkpoint).to(device)
        bio_model = model_class(768 * 10 + features.shape[1]).to(device)
        # 设置optimizer和criterion
        train_bert_params = [id(bert_model.pooler.dense.bias), id(bert_model.pooler.dense.weight)]
        bert_params = filter(lambda p: id(p) not in train_bert_params, bert_model.parameters())
        for param in bert_params:  # 如果需要训练Bert参数则注释这两行
            param.requires_grad = False
        optimizer = optim.Adam(
            [
                # {"params": bert_params, "lr": 1e-6},  # 如果需要训练Bert参数则将这一行取消注释
                {"params": bert_model.pooler.dense.bias, "lr": 1e-3},
                {"params": bert_model.pooler.dense.weight, "lr": 1e-3},
                {"params": bio_model.parameters(), "lr": 1e-3}, 
            ], lr=1e-3
        )

        criterion = nn.MSELoss()

        # 开始训练
        bert_model.train()
        bio_model.train()
        train_epochs = 100
        loss_track = []
        for epoch in tqdm(range(train_epochs), total=train_epochs):
            epoch_pointer = 0
            loss_track_epoch = []
            for epoch_input_ids, epoch_attention_mask, epoch_token_type_ids, features_epoch, labels_epoch in train_dataloader:
                optimizer.zero_grad()

                epoch_input_ids = epoch_input_ids.to(device)
                epoch_attention_mask = epoch_attention_mask.to(device)
                epoch_token_type_ids = epoch_token_type_ids.to(device)

                bert_output = bert_model(input_ids=epoch_input_ids, attention_mask=epoch_attention_mask, token_type_ids=epoch_token_type_ids).last_hidden_state.view(features_epoch.shape[0], -1)  # 将embed结果铺平
                bio_input = torch.cat([bert_output, features_epoch.to(device)], dim=1)

                bio_output = bio_model(bio_input)

                loss = criterion(bio_output.view(labels_epoch.size()), labels_epoch.to(device))  # 在label只有一个特征时需要调整tensor结构
                loss.backward()

                optimizer.step()

                loss_track_epoch.append(loss.detach().to("cpu").item())

            # 训练时验证
            bert_model.eval()
            bio_model.eval()
            pred = []
            with torch.no_grad():
                for epoch_input_ids, epoch_attention_mask, epoch_token_type_ids, features_epoch, labels_epoch in test_dataloader:
                    epoch_input_ids = epoch_input_ids.to(device)
                    epoch_attention_mask = epoch_attention_mask.to(device)
                    epoch_token_type_ids = epoch_token_type_ids.to(device)
                    bert_output = bert_model(input_ids=epoch_input_ids, attention_mask=epoch_attention_mask, token_type_ids=epoch_token_type_ids).last_hidden_state.view(features_epoch.shape[0], -1)
                    bio_input = torch.cat([bert_output, features_epoch.to(device).float()], dim=1)

                    bio_output = bio_model(bio_input)
                    pred.append(bio_output.to("cpu"))
            epoch_pred = torch.cat(pred, dim=0).detach().numpy()
            epoch_true = test_y.to("cpu").numpy()
            validation[epoch, 0, test_id, :] = epoch_pred
            validation[epoch, 1, test_id, :] = epoch_true
            epoch_pointer += epoch_pred.shape[0]

            bert_model.train()
            bio_model.train()

            avg_loss = np.average(loss_track_epoch)
            loss_track.append(avg_loss)
        
        print("Fold {} finished".format(i))
    return validation

def get_rg_error(validation: np.array) -> np.array:
    # 基于训练时保存的结果，计算mse、mae、rmse，会计算每个epoch的分数
    mse_score = np.zeros((100, 18))  # 返回的precision
    mae_score = np.zeros((100, 18))
    pred = validation[:, 0, :, :]
    true = validation[:, 1, :, :]
    for epoch_i in range(100):
        for mmp_i in range(18):
            mse_score[epoch_i, mmp_i] = mean_squared_error(true[epoch_i, :, mmp_i], pred[epoch_i, :, mmp_i])
            mae_score[epoch_i, mmp_i] = mean_absolute_error(true[epoch_i, :, mmp_i], pred[epoch_i, :, mmp_i])
    rg_error = np.stack([mse_score, mae_score, np.sqrt(mse_score)], axis=2)
    return rg_error

def get_cl_error(validate_dl: np.array, threshold: int=1.65) -> np.array:
    # 基于训练时保存的验证结果，计算auc、f1、precision、recall，会计算每个epoch的分数
    pred = validate_dl[:, 0, :, :] > threshold
    true = validate_dl[:, 1, :, :] > 1.65
    p_score = np.zeros((100, 18))
    r_score = np.zeros((100, 18))
    f_score = np.zeros((100, 18))
    auc_score = np.zeros((100, 18))
    for epoch_i in range(100):
        for mmp_i in range(18):
            p_score[epoch_i, mmp_i] = precision_score(true[epoch_i, :, mmp_i], pred[epoch_i, :, mmp_i])
            r_score[epoch_i, mmp_i] = recall_score(true[epoch_i, :, mmp_i], pred[epoch_i, :, mmp_i])
            f_score[epoch_i, mmp_i] = f1_score(true[epoch_i, :, mmp_i], pred[epoch_i, :, mmp_i])
            auc_score[epoch_i, mmp_i] = roc_auc_score(true[epoch_i, :, mmp_i], pred[epoch_i, :, mmp_i])
    cl_error = np.stack([auc_score, f_score, p_score, r_score], axis=2)
    return cl_error


bionn_validation = validate_dl(BioNN, peptides, features_x, all_mmp_y)
bionn_rg_error, bionn_cl_error = get_rg_error(bionn_validation)[-1, :, :], get_cl_error(bionn_validation, threshold=1.65)[-1, :, :]
np.save("./Result/BioNN_rg_error.npy", bionn_rg_error)
np.save("./Result/BioNN_cl_error.npy", bionn_cl_error)

biodnn_validation = validate_dl(BioDeepNN, peptides, features_x, all_mmp_y)
biodnn_rg_error, biodnn_cl_error = get_rg_error(biodnn_validation)[-1, :, :], get_cl_error(biodnn_validation, threshold=1.65)[-1, :, :]
np.save("./Result/BioDNN_rg_error.npy", biodnn_rg_error)
np.save("./Result/BioDNN_cl_error.npy", biodnn_cl_error)

biores_validation = validate_dl(BioResNet, peptides, features_x, all_mmp_y)
biores_rg_error, biores_cl_error = get_rg_error(biores_validation)[-1, :, :], get_cl_error(biores_validation, threshold=1.65)[-1, :, :]
np.save("./Result/BioRes_rg_error.npy", biores_rg_error)
np.save("./Result/BioRes_cl_error.npy", biores_cl_error)

# BioNN
# 100 0.71 (without knn)
# 300 0.70
# 100 0.72 (without cksaap)
# 0.78
# BioDeepNN
# 100 0.6872 (without knn)
# BioResNet
# 100 0.73 (without knn)