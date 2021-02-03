import torch

from scipy import io
from torch.utils.data import DataLoader


test_data = io.loadmat('test.mat')['Testwindow']
test_label = io.loadmat('test.mat')['TestLabel'].ravel()

##
test_image = test_data.reshape([len(test_label), 1, -1, 6])
inp = test_data.shape[1]

tensor_test_data = torch.from_numpy(test_data)
tensor_test_image = torch.from_numpy(test_image)
tensor_label_data = torch.from_numpy(test_label)

test_dataset = list(zip(tensor_test_data, tensor_label_data))
test_imgset = list(zip(tensor_test_image, tensor_label_data))

test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
test_img_loader = DataLoader(dataset=test_imgset, batch_size=len(test_imgset), shuffle=False)

ANNet = torch.load("./ANN.pt")
CNNet = torch.load("./CNN.pt")

Acc1, pred_labels_ann, real_labels1 = ANNet.test(test_data_loader)
Acc2, pred_labels_cnn, real_labels2 = CNNet.test(test_img_loader)

print("ANN model predict accuracy is {}".format(Acc1))
print("CNN model predict accuracy is {}".format(Acc2))

io.savemat('pred_labels_ann.mat', {'pred_labels_ann': pred_labels_ann})
io.savemat('pred_labels_cnn.mat', {'pred_labels_cnn': pred_labels_cnn})