import torch
from torch import nn


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelClassifier, self).__init__()
        hidden_input_size1 = 256  # input_size // 2
        hidden_input_size2 = 128  # input_size // 4

        self.linear = nn.Linear(input_size, hidden_input_size1)
        self.linear2 = nn.Linear(hidden_input_size1, hidden_input_size2)
        self.linear3 = nn.Linear(hidden_input_size2, hidden_input_size2)
        self.linear4 = nn.Linear(hidden_input_size2, output_size)
        self.drop = nn.Dropout(0.1)
        self.act = torch.nn.LeakyReLU()
        # self.act2 = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        # out = self.drop(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear4(out)
        # out = self.act(out)
        return out
