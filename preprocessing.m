clear, clc, close all
%%
data1 = readmatrix(["./data/S2/trial1.csv"]); label1 = load("S2_trial1_label.txt");
data2 = readmatrix(["./data/S2/trial2.csv"]); label2 = load("S2_trial2_label.txt");
data3 = readmatrix(["./data/S1/trial1.csv"]); label3 = load("S1_trial1_label.txt");
data4 = readmatrix(["./data/S1/trial2.csv"]); label4 = load("S1_trial2_label.txt");

X1 = data1(:,14:19)./max(abs(data1(:,14:19)));
X2 = data3(:,14:19)./max(abs(data3(:,14:19)));
X3 = data4(:,14:19)./max(abs(data4(:,14:19)));

Test = data2(:,14:19)./max(abs(data2(:,14:19)));

n = 100;
%% Train
l1 = length(X1);
l2 = length(X2);
l3 = length(X3);
X1window = zeros(l1-2*n,6*(n+1));
X2window = zeros(l2-2*n,6*(n+1));
X3window = zeros(l3-2*n,6*(n+1));

for i = n+1:l1-n
    for j = i-n:i+n
        X1window(i-n,6*(j-i+n)+1:6*(j-i+n+1)) = X1(j,:);
    end
end
for i = n+1:l2-n
    for j = i-n:i+n
        X2window(i-n,6*(j-i+n)+1:6*(j-i+n+1)) = X2(j,:);
    end
end
for i = n+1:l3-n
    for j = i-n:i+n
        X3window(i-n,6*(j-i+n)+1:6*(j-i+n+1)) = X3(j,:);
    end
end
L1window = label1(n+1:l1-n);
L2window = label3(n+1:l2-n);
L3window = label4(n+1:l3-n);

Trainwindow = [X1window; X2window; X3window];
TrainLabelwindow = [L1window; L2window; L3window];

save train.txt Trainwindow -ascii -tabs -double
save train_label.txt TrainLabelwindow -ascii -tabs -double
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