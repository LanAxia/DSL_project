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

class LSTMFilter(nn.Module):
    # 定义LSTM模型
    def __init__(self, input_dim: int = 768, target_dim: int = 16) -> None:
        super(LSTMFilter, self).__init__()
        self.lstm = nn.LSTM(input_dim, target_dim, batch_first=True, bidirectional=True)  # 输出维度是16*2=32

    def forward(self, x: torch.Tensor):
        output, _ = self.lstm(x)
        output = self.nn(output[:, -1, :])
        return output
    

# 滤波器
# LSTM滤波器
class LSTMFilter(nn.Module):
    # 定义LSTM模型
    def __init__(self, input_dim: int = 768, target_dim: int = 16) -> None:
        super(LSTMFilter, self).__init__()
        self.lstm = nn.LSTM(input_dim, target_dim, batch_first=True, bidirectional=True)  # 输出维度是16*2=32

    def forward(self, x: torch.Tensor):
        output, _ = self.lstm(x)
        return output.reshape(output.size(0), -1)
    

# CNN滤波器
class CNNLayer(nn.Module):
    # 定义CNN层
    def __init__(self, in_channels: int = 8, out_channels: int = 16, kernel_size: int = 7, stride: int = 2) -> None:
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor):
        output = self.cnn(x)
        output = self.pool(output)
        output = self.bn(output)
        return output


class CNNFilter(nn.Module):
    # 定义CNN滤波器
    def __init__(self, in_channels: int = 8, kernel_size: int = 7) -> None:
        super(CNNFilter, self).__init__()
        self.cnn1 = CNNLayer(in_channels, 1, kernel_size, stride=3)
        self.cnn2 = CNNLayer(1, 1, kernel_size, stride=1)
    
    def forward(self, x: torch.Tensor):
        output = self.cnn1(x)
        output = self.cnn2(output)
        output = output.squeeze(1)
        return output


# LSTM编码器
class LSTMEncoder(nn.Module):
    # 定义LSTM编码器
    def __init__(self, input_dim: int = 768, target_dim: int = 128) -> None:
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, target_dim, batch_first=True, bidirectional=True)  # 输出维度是128*2=256
    
    def forward(self, x: torch.Tensor):
        seq, (h, c) = self.lstm(x)
        return torch.cat([h[0], h[1]], dim=1)