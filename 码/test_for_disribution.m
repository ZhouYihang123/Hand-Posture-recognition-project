% test for distribution
clear
clc

filename = 'D_train.csv';
[feature, label, user] = preprocessing(filename, 3, 7, 1);
% preprocessing(filename, pca_dimension = 3, threshold = 7, flag = 1)
map = [ 1 0 0
    0 1 0
    0 0 1
    1 1 0
    1 0 1];
colormap(map);
scatter3(feature(:,1),feature(:,2),feature(:,3),1,label)