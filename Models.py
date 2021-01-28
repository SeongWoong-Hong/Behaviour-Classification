from typing import Type

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def fit(self, train_loader):
        self.train()
        device = next(self.parameters()).device.index
        losses = []
        for i, data in enumerate(train_loader):
            image = data[0].type(torch.FloatTensor).cuda(device)
            label = data[1].type(torch.LongTensor).cuda(device)

            pred_label = self.forward(image)
            loss = self.criterion(pred_label, label)
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def test(self, test_loader):
        self.eval()
        device = next(self.parameters()).device.index
        pred_labels = []
        real_labels = []
        for i, data in enumerate(test_loader):
            image = data[0].type(torch.FloatTensor).cuda(device)
            label = data[1].type(torch.LongTensor).cuda(device)
            real_labels += list(label.cpu().detach().numpy())

            pred_label = self.forward(image)
            pred_label = list(pred_label.cpu().detach().numpy())
            pred_labels += pred_label

        real_labels = np.array(real_labels)
        pred_labels = np.array(pred_labels)
        pred_labels = pred_labels.argmax(axis=1)
        acc = sum(real_labels == pred_labels) / len(real_labels) * 100
        return acc, pred_labels, real_labels


class ANNModel(BaseModel):
    def __init__(self,
                 inp: int,
                 oup: int,
                 lr: float = 1e-4,
                 momentum: float = 0.9,
                 optim: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 criterion=nn.CrossEntropyLoss):
        """
        :param inp: (int) the size of inputs
        :param oup: (int) the number of labels of data
        :param lr: (float) learning rate
        """
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(inp, 128)
        self.layer2 = nn.Linear(128, oup)
        self.relu = nn.ReLU(inplace=True)
        self.optim = optim(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = criterion()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        out = self.relu(self.layer2(x))
        return out


class CNNModel(BaseModel):
    def __init__(self,
                 inp: int,
                 oup: int,
                 lst_ch: int,
                 lr: float = 1e-4,
                 momentum: float = 0.9,
                 optim: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 criterion=nn.CrossEntropyLoss
                 ):
        """
        :param inp: (int) the total number of the input image pixels (channel * width * height)
        :param oup: (int) the number of labels
        :param lr: (float) learning rate
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, lst_ch, kernel_size=3, padding=1)
        self.linear = nn.Linear(inp * lst_ch, oup)
        self.relu = nn.ReLU(inplace=True)
        self.optim = optim(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = criterion()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Not implemented yet
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.len = 1

    def forward(self, x):
        # packed = nn.utils.rnn.pack_padded_sequence(x, self.len)
        output, (hidden, cell) = self.lstm(x)
        out = self.relu(self.fc(hidden.squeeze(0)))
        return out
