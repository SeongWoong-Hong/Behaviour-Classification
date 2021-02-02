# Header
import csv
import torch
import numpy as np

from torch.utils.data import DataLoader
from Models import ANNModel, CNNModel, RNNModel
from matplotlib import pyplot as plt


# hyper parameters for learning
BATCH_SIZE = 512

# Data Loading - test.txt, test_label.txt, train.txt, train_label.txt should be contained in the same directory
traindata = csv.reader(open("train.txt"), delimiter='\t')
trainlabel = csv.reader(open("train_label.txt"), delimiter='\t')
valdata = csv.reader(open("validation.txt"), delimiter='\t')
vallabel = csv.reader(open("val_label.txt"), delimiter='\t')

train_label = []
train_data = []
val_label = []
val_data = []

for line in trainlabel:
    train_label.append(int(float(line[0])))

for line in traindata:
    train_data.append([float(i) for i in line[0:-1]])

for line in vallabel:
    val_label.append(int(float(line[0])))

for line in valdata:
    val_data.append([float(i) for i in line[0:-1]])

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

val_data = np.array(val_data, dtype=np.float32)
val_image = val_data.reshape([len(val_label), 1, -1, 6])
label_data = np.array(val_label, dtype=np.int)

tensor_val_data = torch.from_numpy(val_data)
tensor_val_image = torch.from_numpy(val_image)
tensor_label_data = torch.from_numpy(label_data)

val_dataset = list(zip(tensor_val_data, tensor_label_data))
val_imgset = list(zip(tensor_val_image, tensor_label_data))

train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_img_loader = DataLoader(dataset=train_imgset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
val_img_loader = DataLoader(dataset=val_imgset, batch_size=len(val_imgset), shuffle=False)


# learning process
ANNet = ANNModel(inp, 4, optim=torch.optim.Adam).cuda()
CNNet = CNNModel(img_size, 4, 4, optim=torch.optim.Adam).cuda()
# RNNet = RNNModel(inp, 128, 4).cuda()  # Not Implemented yet
EpochLoss = []
Acc = []
for epoch in range(500):
    CurrentEpochLoss = ANNet.fit(train_data_loader)
    CurrentAcc, _, _ = ANNet.test(val_data_loader)
    EpochLoss.append(CurrentEpochLoss)
    Acc.append(CurrentAcc)
    if (epoch + 1) % 10 == 0:
        print("ANN model {}th Epoch. Average Loss is {:.5f}. "
              "Test Acc is {:.2f}".format(epoch + 1, CurrentEpochLoss, CurrentAcc))

plt.plot(Acc)
plt.xlabel("epoch")
plt.ylabel("Val. Acc. (%)")
plt.title("ANN Accuracy")
plt.show()

EpochLoss = []
Acc = []
for epoch in range(300):
    CurrentEpochLoss = CNNet.fit(train_img_loader)
    CurrentAcc, _, _ = CNNet.test(val_img_loader)
    EpochLoss.append(CurrentEpochLoss)
    Acc.append(CurrentAcc)
    if (epoch + 1) % 10 == 0:
        print("CNN model {}th Epoch. Average Loss is {:.5f}. "
              "Test Acc is {:.2f}".format(epoch + 1, CurrentEpochLoss, CurrentAcc))

plt.plot(Acc)
plt.xlabel("epoch")
plt.ylabel("Val. Acc. (%)")
plt.title("CNN Accuracy")
plt.show()

torch.save(ANNet, "ANN.pt")
torch.save(CNNet, "CNN.pt")
