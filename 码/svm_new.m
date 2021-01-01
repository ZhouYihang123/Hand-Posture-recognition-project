clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
[feature_train, label_train, user, v, mstd] = preprocessing(filename_train, 16, 7, 0, 0, 3, 0);
[feature_test, label_test, user1, v, mstd] = preprocessing(filename_test, 16, 7, 1, v, 3, mstd);

c=61;g=12.1;
cost = 10^((c/10)-6);
gamma = 10^((g/10)-3);
libsvm_options=['-s 0  -c ',mat2str(cost),' -g ',mat2str(gamma),' -t 2'];
svmmodel = svmtrain(label_train, feature_train, libsvm_options);

[predict_label, accuracy, dec_values] = svmpredict(label_test, feature_test, svmmodel);




%%
clear
filename_train = 'D_train_large.csv';
filename_test = 'D_test.csv';
[feature_train, label_train, user, v, mstd] = preprocessing_svm(filename_train, 22, 6, 0, 0, 2, 0);
[feature_test, label_test, user1, v, mstd] = preprocessing_svm(filename_test, 22, 6, 1, v, 2, mstd);
% [feature_train, label_train, user, v, mstd] = preprocessing_svm(filename_train, 16, 6, 0, 0, 3, 0);
% [feature_test, label_test, user1, v, mstd] = preprocessing_svm(filename_test, 16, 6, 1, v, 3, mstd);
c=65.5;g=3.9;
%c=57.7;g=17.3;
cost = 10^((c/10)-6);
gamma = 10^((g/10)-3);
libsvm_options=['-s 0  -c ',mat2str(cost),' -g ',mat2str(gamma),' -t 2'];
svmmodel = svmtrain(label_train, feature_train, libsvm_options);

[predict_label, accuracy, dec_values] = svmpredict(label_test, feature_test, svmmodel);
