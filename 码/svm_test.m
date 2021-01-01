clear
clc

filename = 'D_train.csv';
[feature_train, label_train, user, v, mstd] = preprocessing(filename, 16, 7, 0, 0, 3, 0);
data=[label_train, feature_train];
acc_mean=zeros(120,60);
acc_std=zeros(60,60);
for c = 1:120
    for g = 1:60
        cost = 10^((c/10)-6);
        gamma = 10^((g/10)-3);
        [acc_mean(c,g),acc_std(c,g)] = crossval_svm(data,c,g);
    end
end
[M,I]=max(max(acc_mean))
% best:(125,1)


% second 
clear
clc

filename = 'D_train.csv';
[feature_train, label_train, user, v, mstd] = preprocessing_svm(filename, 16, 7, 0, 0, 3, 0);
data=[label_train, feature_train];
acc_mean=zeros(120,60);
acc_std=zeros(60,60);
for c = 1:120
    for g = 1:60
        cost = 10^((c/10)-6);
        gamma = 10^((g/10)-3);
        [acc_mean(c,g),acc_std(c,g)] = crossval_svm(data,c,g);
    end
end
[M,I]=max(max(acc_mean))
% best:(125,1)