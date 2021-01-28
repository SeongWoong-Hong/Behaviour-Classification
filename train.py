# Header
import csv
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from Models import ANNModel, CNNModel, RNNModel
from matplotlib import pyplot as plt


# hyper parameters for learning
BATCH_SIZE = 128

# Data Loading - test.txt, test_label.txt, train.txt, train_label.txt should be contained in the same directory
traindata = csv.reader(open("train.txt"), delimiter='\t')
trainlabel = csv.reader(open("train_label.txt"), delimiter='\t')
testdata = csv.reader(open("test.txt"), delimiter='\t')
testlabel = csv.reader(open("test_label.txt"), delimiter='\t')

train_label = []
train_data = []
test_label = []
test_data = []

for line in trainlabel:
    train_label.append(int(float(line[0])))

for line in traindata:
    train_data.append([float(i) for i in line[0:-1]])

for line in testlabel:
    test_label.append(int(float(line[0])))

for line in testdata:
    test_data.append([float(i) for i in line[0:-1]])

# Set the train, test dataset
train_data = np.array(train_data, dtype=np.float32)
train_image = train_data.reshape([len(train_label), 1, -1, 6])
img_size = train_image.shape[1]*train_image.shape[2]*train_image.shape[3]
label_data = np.array(train_label, dtype=np.int)
inp = train_data.shape[1]

tensor_train_data = torch.from_numpy(train_data)
tensor_train_image = torch.from_numpy(train_image)
tensor_label_data = torch.from_numpy(label_data)

train_dataset = list(zip(tensor_train_data, tensor_label_data))
train_imgset = list(zip(tensor_train_image, tensor_label_data))

test_data = np.array(test_data, dtype=np.float32)
test_image = test_data.reshape([len(test_label), 1, -1, 6])
label_data = np.array(test_label, dtype=np.int)

tensor_test_data = torch.from_numpy(test_data)
tensor_test_image = torch.from_numpy(test_image)
tensor_label_data = torch.from_numpy(label_data)

test_dataset = list(zip(tensor_test_data, tensor_label_data))
test_imgset = list(zip(tensor_test_image, tensor_label_data))

train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_img_loader = DataLoader(dataset=train_imgset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_img_loader = DataLoader(dataset=test_imgset, batch_size=BATCH_SIZE, shuffle=False)


# learning process
ANNet = ANNModel(inp, 4, optim=torch.optim.SGD).cuda()
CNNet = CNNModel(img_size, 4, 4, optim=torch.optim.SGD).cuda()
# RNNet = RNNModel(inp, 128, 4).cuda()  # Not Implemented yet
EpochLoss = []
Acc = []
for epoch in range(200):
    CurrentEpochLoss = ANNet.fit(train_data_loader)
    CurrentAcc, _, _ = ANNet.test(test_data_loader)
    print("ANN.pt {}th Epoch. Average Loss is {:.5f}. Test Acc is {:.2f}".format(epoch+1, CurrentEpochLoss, CurrentAcc))
    EpochLoss.append(CurrentEpochLoss)
    Acc.append(CurrentAcc)
for epoch in range(200):
    CurrentEpochLoss = CNNet.fit(train_img_loader)
    CurrentAcc, _, _ = CNNet.test(test_img_loader)
    print("CNN.pt {}th Epoch. Average Loss is {:.5f}. Test Acc is {:.2f}".format(epoch+1, CurrentEpochLoss, CurrentAcc))
    EpochLoss.append(CurrentEpochLoss)
    Acc.append(CurrentAcc)

torch.save(ANNet.state_dict(), "ANN.pt")
torch.save(CNNet.state_dict(), "CNN.pt")
