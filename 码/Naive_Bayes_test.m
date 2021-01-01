%% Naive Bayes 89%
clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
[feature_train, label_train, user, v, mstd] = preprocessing(filename_train, 10, 7, 0, 0, 0, 0);
[feature_test, label_test, user1, v, mstd] = preprocessing(filename_test, 10, 7, 1, v, 0, mstd);
[V,~] = eig(feature_train'*feature_train);
% Conv=cov(feature_train);
% [V,~]=eig(Conv);
feature_train = feature_train*V;
feature_test = feature_test*V;
Conv=cov(feature_train);
imagesc(Conv)

model_nb = fitcnb(feature_train, label_train);
C = model_nb.predict(feature_test);
result_nbayes = find(label_test-C~=0);
acc_nbates = 1-length(result_nbayes)/length(label_test);

%%
%% Naive Bayes 2 90%
clear
filename_train = 'D_train_large.csv';
filename_test = 'D_test.csv';
pca=13;

[feature_train, label_train, user, v, mstd] = preprocessing_svm(filename_train, pca, 6, 0, 0, 1, 0);
[feature_test, label_test, user1, v, mstd] = preprocessing_svm(filename_test, pca, 6, 1, v, 1, mstd);
[V,D] = eig(feature_train'*feature_train);
Conv=cov(feature_train);
imagesc(Conv)
% [V,~]=eig(Conv);
feature_train = feature_train*V;
feature_test = feature_test*V;
Conv=cov(feature_train);
imagesc(Conv)

% model_nb = fitcnb(feature_train(:,ceil(pca/2)-1:pca), label_train);
% C = model_nb.predict(feature_test(:,ceil(pca/2)-1:pca));
start=4;
model_nb = fitcnb(feature_train(:,start:pca), label_train);
C = model_nb.predict(feature_test(:,start:pca));
result_nbayes = find(label_test-C~=0);
acc_nbates = 1-length(result_nbayes)/length(label_test);
%plotconfusion(label_test,C);
%% Random classifier
random_label=randi(5,17059,1);
result_nbayes = find(label_test-random_label~=0);
acc_random = 1-length(result_nbayes)/length(label_test);
%% Naive Bayes 3 84%
clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
[feature_train, label_train, user, v, mstd] = preprocessing_svm(filename_train, 19, 7, 0, 0, 3, 0);
[feature_test, label_test, user1, v, mstd] = preprocessing_svm(filename_test, 19, 7, 1, v, 3, mstd);
[V,~] = eig(feature_train'*feature_train);
%Conv=cov(feature_train);
%imagesc(Conv)
% [V,~]=eig(Conv);
feature_train = feature_train*V;
feature_test = feature_test*V;
Conv=cov(feature_train);
imagesc(Conv)
model_nb = fitcnb(feature_train(:,3:19), label_train);
C = model_nb.predict(feature_test(:,3:19));
result_nbayes = find(label_test-C~=0);
acc_nbates = 1-length(result_nbayes)/length(label_test);