# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# define Bert-Bio Model
class BioNN(nn.Module):
    def __init__(self, hidden_size: int = 768, labels_num: int = 18) -> None:
        super(BioNN, self).__init__()
        self.nn1 = nn.Linear(hidden_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.nn2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.nn3 = nn.Linear(256, labels_num)

    def forward(self, x: torch.Tensor):
        output = F.tanh(self.nn1(x))
        output = self.dropout1(output)
        output = F.tanh(self.nn2(output))
        output = self.dropout2(output)
        output = self.nn3(output)
        return output


class BioDeepNN(nn.Module):
    # 深度神经网络，在原来的基础上增加了模型的深度
    def __init__(self, hidden_size: int = 768, labels_num: int = 18) -> None:
        super(BioDeepNN, self).__init__()
        self.nn1 = nn.Linear(hidden_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.nn2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.nn3 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.nn4 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(0.5)
        self.nn5 = nn.Linear(256, labels_num)

    def forward(self, x: torch.Tensor):
        output = F.tanh(self.nn1(x))
        output = self.dropout1(output)
        output = F.tanh(self.nn2(output))
        output = self.dropout2(output)
        output = F.tanh(self.nn3(output))
        output = self.dropout3(output)
        output = F.tanh(self.nn4(output))
        output = self.dropout4(output)
        output = self.nn5(output)
        return output


class ResidualBlock(nn.Module):
    # 定义残差
    def __init__(self, hidden_size: int = 256) -> None:
        super(ResidualBlock, self).__init__()
        self.nn1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.nn2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor):
        output = F.tanh(self.nn1(x))
        output = self.dropout1(output)
        output = F.tanh(self.nn2(output))
        output = output + x
        return output


class BioResNet(nn.Module):
    # 深度神经网络，在原来的基础上增加了模型的深度
    def __init__(self, hidden_size: int = 768, labels_num: int = 18) -> None:
        super(BioResNet, self).__init__()
        self.nn1 = nn.Linear(hidden_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.nn2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.nn3 = ResidualBlock(256)
        self.dropout3 = nn.Dropout(0.5)
        self.nn4 = ResidualBlock(256)
        self.dropout4 = nn.Dropout(0.5)
        self.nn5 = nn.Linear(256, labels_num)

    def forward(self, x: torch.Tensor):
        output = F.tanh(self.nn1(x))
        output = self.dropout1(output)
        output = F.tanh(self.nn2(output))
        output = self.dropout2(output)
        output = F.tanh(self.nn3(output))
        output = self.dropout3(output)
        output = F.tanh(self.nn4(output))
        output = self.dropout4(output)
        output = self.nn5(output)
        return output
