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
