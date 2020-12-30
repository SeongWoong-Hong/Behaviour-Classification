import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, inp, oup):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(inp, 128)
        self.layer2 = nn.Linear(128, oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        out = self.relu(self.layer2(x))
        return out

class CNNModel(nn.Module):
    #inp: total number of input image pixels. (channel * width * height)
    #oup: total number of labels.
    def __init__(self, inp, oup):
        super(CNNModel, self).__init__()
        lst_ch = 4
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, lst_ch, kernel_size=3, padding=1)
        self.linear = nn.Linear(inp * lst_ch, oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.len = 1

    def forward(self, x):
        packed = nn.utils.rnn.pack_padded_sequence(x, self.len)
        output, (hidden, cell) = self.lstm(x)
        out = self.relu(self.fc(hidden.squeeze(0)))
        return out