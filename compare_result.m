clc, clear, close all
%%
ann_pred_label = load("pred_labels_ann.txt");
cnn_pred_label = load("pred_labels_cnn.txt");
test_label = load("test_label.txt");
test_data = load("test.txt");

ANN_accuray = sum(ann_pred_label==test_label)/length(ann_pred_label);
CNN_accuray = sum(cnn_pred_label==test_label)/length(cnn_pred_label);

labels = {test_label, ann_pred_label, cnn_pred_label};
data = test_data(:, 2);
for i = 1:length(labels)
    label = labels{i};
    s = find(label==0);
    w = find(label==1);
    sw = find(label==2);
    t = find(label==3);
    figure; hold on
    scatter(s, data(s), '.');
    scatter(w, data(w), '.');
    scatter(sw, data(sw), '.');
    scatter(t, data(t), '.');
    legend("Stay", "Walking", "Swing", "Transition")
end

fprintf("\nANN model prediction accuray is %.2f\n", ANN_accuray)
fprintf("\nCNN model prediction accuray is %.2f\n", CNN_accuray)