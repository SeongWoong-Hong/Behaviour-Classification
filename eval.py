import csv
import torch

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

testdata = csv.reader(open("test.txt"), delimiter='\t')
testlabel = csv.reader(open("test_label.txt"), delimiter='\t')

test_label = []
test_data = []

for line in testlabel:
    test_label.append(int(float(line[0])))

for line in testdata:
    test_data.append([float(i) for i in line[0:-1]])


def eval(model, test_loader):
    model.eval()
    device = next(model.parameters()).device.index
    pred_labels = []
    real_labels = []
    for i, data in enumerate(test_loader):
        image = data[0].type(torch.FloatTensor).cuda(device)
        label = data[1].type(torch.LongTensor).cuda(device)
        real_labels += list(label.cpu().detach().numpy())

        pred_label = model(image)
        pred_label = list(pred_label.cpu().detach().numpy())
        pred_labels += pred_label

    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.argmax(axis=1)
    acc = sum(real_labels == pred_labels) / len(real_labels) * 100
    return acc, pred_labels, real_labels


##
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
CRITERION = nn.CrossEntropyLoss()

##
test_data = np.array(test_data, dtype=np.float32)
test_image = test_data.reshape([len(test_label), 1, -1, 6])
label_data = np.array(test_label, dtype=np.int)
inp = test_data.shape[1]

tensor_test_data = torch.from_numpy(test_data)
tensor_test_image = torch.from_numpy(test_image)
tensor_label_data = torch.from_numpy(label_data)

test_dataset = list(zip(tensor_test_data, tensor_label_data))
test_imgset = list(zip(tensor_test_image, tensor_label_data))

test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_img_loader = DataLoader(dataset=test_imgset, batch_size=BATCH_SIZE, shuffle=False)

ANNet = torch.load("./ANN.pt")
CNNet = torch.load("./CNN.pt")

Acc1, pred_labels_ann, real_labels1 = eval(ANNet, test_data_loader)
Acc2, pred_labels_cnn, real_labels2 = eval(CNNet, test_img_loader)

f1 = open("pred_labels_ann.txt", "w", encoding='ascii', newline='')
f2 = open("pred_labels_cnn.txt", "w", encoding='ascii', newline='')
wr1 = csv.writer(f1)
wr2 = csv.writer(f2)

for param in pred_labels_ann:
    wr1.writerow([param, ''])
f1.close()

for param in pred_labels_cnn:
    wr2.writerow([param, ''])
f2.close()