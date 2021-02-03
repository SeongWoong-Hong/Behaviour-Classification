clc, clear, close all
%%
ann_pred_label = load("pred_labels_ann.txt");
cnn_pred_label = load("pred_labels_cnn.txt");
test_label = load("test_label.txt");
test_data = load("test.txt");

ANN_accuracy = 100*sum(ann_pred_label==test_label)/length(ann_pred_label);
CNN_accuracy = 100*sum(cnn_pred_label==test_label)/length(cnn_pred_label);

for i = 0:3
    ann_acc(i+1) = sum(ann_pred_label(test_label==i)==i)/sum(test_label==i);
    cnn_acc(i+1) = sum(cnn_pred_label(test_label==i)==i)/sum(test_label==i);
end

figure;
cm = confusionmat(test_label, ann_pred_label);
cc = confusionchart(cm, {'Stay','Walking','Swing','Trans.'});
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'column-normalized';
title("Confusion Matrix - ANN model")

figure;
cm = confusionmat(test_label, cnn_pred_label);
cc = confusionchart(cm, {'Stay','Walking','Swing','Trans.'});
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'column-normalized';
title("Confusion Matrix - CNN model")


labels = {test_label, ann_pred_label, cnn_pred_label};
titles = ["Reference Signal", ...
            "ANN model prediction", ...
            "CNN model prediction"];
data = test_data(:, 3);
for i = 1:length(labels)
    label = labels{i};
    s = find(label==0);
    w = find(label==1);
    sw = find(label==2);
    t = find(label==3);
    figure; hold on
    scatter(s/100, data(s), '.');
    scatter(w/100, data(w), '.');
    scatter(sw/100, data(sw), '.');
    scatter(t/100, data(t), '.');
    legend("Stay", "Walking", "Swing", "Transition")
    title(titles(i))
    xlabel("time(s)")
    ylabel("gyro_y")
end

fprintf("\nANN model prediction accuray is %.2f\n", ANN_accuracy)
fprintf("\nCNN model prediction accuray is %.2f\n", CNN_accuracy)