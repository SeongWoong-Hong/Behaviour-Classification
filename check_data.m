clear, clc, close all
%%
data1 = readmatrix("./data/S1/trial1.csv"); label1 = load("./data/S1_trial1_label.txt");
data2 = readmatrix("./data/S1/trial2.csv"); label2 = load("./data/S1_trial2_label.txt");
data3 = readmatrix("./data/S2/trial1.csv"); label3 = load("./data/S2_trial1_label.txt");
data4 = readmatrix("./data/S2/trial2.csv"); label4 = load("./data/S2_trial2_label.txt");
data5 = readmatrix("./data/S3/trial1.csv"); label5 = load("./data/S3_trial1_label.txt");
data6 = readmatrix("./data/S3/trial2.csv"); label6 = load("./data/S3_trial2_label.txt");

labels = {label1, label2, label3, label4, label5, label6};
datas = {data1, data2, data3, data4, data5, data6};

for i = 1:length(labels)
    label = labels{i}; data = datas{i}(:, 15);
    s = find(label==0);
    w = find(label==1);
    sw = find(label==2);
    t = find(label==3);
    figure; hold on
    scatter(s, data(s), '.');
    scatter(w, data(w), '.');
    scatter(sw, data(sw), '.');
    scatter(t, data(t), '.');
end