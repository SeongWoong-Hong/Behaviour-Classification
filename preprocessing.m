clear, clc, close all
%%
data1 = readmatrix("./data/S1/trial1.csv"); label1 = load("./data/S1_trial1_label.txt");
data2 = readmatrix("./data/S1/trial2.csv"); label2 = load("./data/S1_trial2_label.txt");
data3 = readmatrix("./data/S2/trial1.csv"); label3 = load("./data/S2_trial1_label.txt");
data4 = readmatrix("./data/S2/trial2.csv"); label4 = load("./data/S2_trial2_label.txt");
data5 = readmatrix("./data/S3/trial1.csv"); label5 = load("./data/S3_trial1_label.txt");
data6 = readmatrix("./data/S3/trial2.csv"); label6 = load("./data/S3_trial2_label.txt");

% 14th to 19th columns are acc and gyro measurements
X1 = data1(:,14:19)./max(abs(data1(:,14:19)));
X2 = data3(:,14:19)./max(abs(data3(:,14:19)));
X3 = data4(:,14:19)./max(abs(data4(:,14:19)));
X4 = data5(:,14:19)./max(abs(data5(:,14:19)));

Train = {X1, X2, X3, X4};
labels = {label1, label3, label4, label5, label6};
Val = data6(:,14:19)./max(abs(data6(:,14:19)));
Test = data2(:,14:19)./max(abs(data2(:,14:19)));

n = 100;
%% Train
Trainwindow = []; TrainLabelwindow = [];
for k = 1:4
    X = Train{k}; label = labels{k};
    l = length(X);
    Xwindow = zeros(l-2*n, 6*(n+1));
    for i = n+1:l-n
        for j = i-n:i+n
            Xwindow(i-n,6*(j-i+n)+1:6*(j-i+n+1)) = X(j,:);
        end
    end
    Lwindow = label(n+1:l-n);
    Trainwindow = [Trainwindow; Xwindow];
    TrainLabelwindow = [TrainLabelwindow; Lwindow];
end

save train.txt Trainwindow -ascii -tabs -double
save train_label.txt TrainLabelwindow -ascii -tabs -double
%% Validation
l = length(Val);
Valwindow = zeros(l-2*n, 6*(n+1));
for i = n+1:l-n
    for j = i-n:i+n
        Valwindow(i-n,6*(j-i+n)+1:6*(j-i+n+1)) = Val(j,:);
    end
end
ValLabel = label6(n+1:l-n);

save validation.txt Valwindow -ascii -tabs -double
save val_label.txt ValLabel -ascii -tabs -double
%% Test
l = length(Test);
Testwindow = zeros(l-2*n, 6*(n+1));
for i = n+1:l-n
    for j = i-n:i+n
        Testwindow(i-n,6*(j-i+n)+1:6*(j-i+n+1)) = Test(j,:);
    end
end
TestLabel = label2(n+1:l-n);

save test.txt Testwindow -ascii -tabs -double
save test_label.txt TestLabel -ascii -tabs -double