# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizerFast, BertModel


# 导入其他文件
from extract_features import load_features
from models import BioNN, BioDeepNN, BioResNet

# constant
SAVE = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 定义Dataset和DataLoader
class PeptidesDataset(Dataset):
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
    def __init__(self, peptides, features, labels, batch_size, shuffle=True):
        dataset = PeptidesDataset(peptides, features, labels)
        super().__init__(dataset, batch_size, shuffle)


if __name__ == "__main__":
    # import data and preprocess
    data = pd.read_csv("./Data/processed_peptides10.csv")  # load data

    # 得到氨基酸序列
    peptides = data.iloc[:, 0].values.tolist()  # 肽链的列表（字符串）

    # load extracted features
    features_x = load_features().iloc[:, 1:].values
    features_x = torch.from_numpy(features_x).float()  # 转换为tensor

    # import Bert-Base-Protein model
    checkpoint = 'unikei/bert-base-proteins'
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    bert_model = BertModel.from_pretrained(checkpoint).to(device)

    # 处理mmp y的数据
    labels = data.iloc[:, 1:].values
    regular_coefficient = np.max(np.abs(labels))
    labels = labels / regular_coefficient  # 归一化处理
    labels = torch.from_numpy(labels).float().to(device)

    # 划分训练集和测试集
    train_peptides, test_peptides, train_features, test_features, train_labels, test_labels = train_test_split(peptides,
                                                                                                               features_x,
                                                                                                               labels,
                                                                                                               test_size=0.2,
                                                                                                               random_state=33)
    train_dataloader = PeptidesDataLoader(train_peptides, train_features, train_labels, 512, shuffle=True)
    test_dataloader = PeptidesDataLoader(test_peptides, test_features, test_labels, 512, shuffle=False)

    # 创建全连接模型
    bio_model = BioDeepNN(768 * 10 + features_x.shape[1], labels_num=18).to(device)  # 预测全部mmp

    # 训练模型
    # 设置optimizer
    train_bert_params = [id(bert_model.pooler.dense.bias), id(bert_model.pooler.dense.weight)]
    bert_params = filter(lambda p: id(p) not in train_bert_params, bert_model.parameters())
    optimizer = optim.Adam(
        [
            {"params": bert_params, "lr": 1e-6},
            {"params": bert_model.pooler.dense.bias, "lr": 1e-3},
            {"params": bert_model.pooler.dense.weight, "lr": 1e-3},
            {"params": bio_model.parameters(), "lr": 1e-3}
        ], lr=1e-3
    )

    # 设置criterion
    criterion = nn.MSELoss()

    # 调整为train模式
    bert_model.train()
    bio_model.train()

    # 开始训练
    train_epochs = 100
    pbar = tqdm(range(train_epochs), desc="Training: ", total=train_epochs)
    pbar.set_postfix(loss=0)
    loss_track = []
    for epoch in pbar:
        loss_track_epoch = []
        for peptides_epoch, features_epoch, labels_epoch in train_dataloader:
            optimizer.zero_grad()

            tokens_epoch = tokenizer(peptides_epoch, return_tensors='pt').to(device)
            bert_output = bert_model(**tokens_epoch).last_hidden_state.view(len(peptides_epoch), -1)  # 将embed结果铺平
            bio_input = torch.cat([bert_output, features_epoch.to(device)], dim=1)

            bio_output = bio_model(bio_input)

            loss = criterion(bio_output.view(labels_epoch.size()), labels_epoch)  # 在label只有一个特征时需要调整tensor结构
            loss.backward()

            optimizer.step()

            loss_track_epoch.append(loss.detach().to("cpu").item())

        avg_loss = np.average(loss_track_epoch)
        loss_track.append(avg_loss)
        pbar.set_postfix(loss=avg_loss)

    # 测试模型效果
    bert_model.eval()
    bio_model.eval()

    # 预测回归结果
    test_pred = []
    with torch.no_grad():
        for peptides_epoch, features_epoch, labels_epoch in test_dataloader:
            tokens_epoch = tokenizer(peptides_epoch, return_tensors='pt').to(device)
            bert_output = bert_model(**tokens_epoch).last_hidden_state.view(len(peptides_epoch), -1)  # 将embed结果铺平
            bio_input = torch.cat([bert_output, features_epoch.to(device)], dim=1)

            bio_output = bio_model(bio_input)
            test_pred.append(bio_output.to("cpu"))
    test_pred = torch.cat(test_pred, dim=0).detach().numpy() * regular_coefficient
    test_truth = test_labels.to("cpu").numpy() * regular_coefficient

    # 保存mse
    mse = np.zeros(labels.shape[1])
    for i in range(labels.shape[1]):
        mse[i] = np.sqrt(sklearn.metrics.mean_squared_error(test_truth[:, i], test_pred[:, i]))
    mse = pd.DataFrame(mse)
    mse.to_csv("./Cache/scores_bionn.csv", index=False)

    # 保存模型
    torch.save(bert_model.state_dict(), "./Model/bert_model.pth")
    torch.save(bio_model.state_dict(), "./Model/bio_model.pth")

    print(np.mean(mse))
    # BioNN
    # 100 0.71 (without knn)
    # 300 0.70
    # 100 0.72 (without cksaap)
    # 0.78
    # BioDeepNN
    # 100 0.6872 (without knn)
    # BioResNet
    # 100 0.73 (without knn)
