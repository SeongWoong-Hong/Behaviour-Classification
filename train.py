# Header
import torch

from scipy import io
from torch.utils.data import DataLoader
from Models import ANNModel, CNNModel, RNNModel


# hyper parameters for learning
BATCH_SIZE = 512

# Data Loading - test.txt, test_label.txt, train.txt, train_label.txt should be contained in the same directory
train_data = io.loadmat('train.mat')['Trainwindow']
train_label = io.loadmat('train.mat')['TrainLabelwindow'].ravel()
val_data = io.loadmat('validation.mat')['Valwindow']
val_label = io.loadmat('validation.mat')['ValLabel'].ravel()

# Set the train, test dataset
train_image = train_data.reshape([len(train_label), 1, -1, 6])
val_image = val_data.reshape([len(val_label), 1, -1, 6])
img_size = train_image.shape[1]*train_image.shape[2]*train_image.shape[3]
inp = train_data.shape[1]

tensor_train_data = torch.from_numpy(train_data)
tensor_train_image = torch.from_numpy(train_image)
tensor_label_data = torch.from_numpy(train_label)

train_dataset = list(zip(tensor_train_data, tensor_label_data))
train_imgset = list(zip(tensor_train_image, tensor_label_data))

tensor_val_data = torch.from_numpy(val_data)
tensor_val_image = torch.from_numpy(val_image)
tensor_label_data = torch.from_numpy(val_label)

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

ANNet.learn(500, train_data_loader, val_data_loader)
torch.save(ANNet, "ANN.pt")

CNNet.learn(500, train_img_loader, val_img_loader)
torch.save(CNNet, "CNN.pt")
