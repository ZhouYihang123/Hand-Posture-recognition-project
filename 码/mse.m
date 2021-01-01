clear
addpath('C:\Users\hang\Desktop\file\ee559\HW6\prtools')
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
threshold=0;
pca_train=0;
pca_test=0;
type=16;
V=0;%dont change
acc=zeros(1,2);
for pca_dim=1


[feature_train, label_train, user,V] = preprocessing_simple(filename_train,threshold,  ...
    pca_train,pca_dim,V,type);
[feature_test, label_test, user1,V] = preprocessing_simple(filename_test,threshold,  ...
    pca_test,pca_dim,V,type);

% %std
% feature_size=length(feature_train(1,:));
% train_number=length(feature_train(:,1));
% test_number=length(feature_test(:,1));
% mean=zeros(feature_size,1);
% sigma=zeros(feature_size,1);
% train_std=zeros(train_number,feature_size);
% test_std=zeros(test_number,feature_size);
% 
% for t=1:feature_size
%     [train_std(:,t),mean(t,1),sigma(t,1)]=zscore(feature_train(:,t));
%     test_std(:,t)=(feature_test(:,t)-mean(t,1))/sigma(t,1);
% end
% feature_train=zscore(feature_train);
% feature_test=test_std;

% %normalization
% large=max(feature_train);
% small=min(feature_train);
% feature_train=(feature_train-small)/(large-small);
% feature_test=(feature_test-small)/(large-small);

hangtr_set=prdataset(feature_train,label_train);%1:2
hangtr_w=fisherc(hangtr_set);
acc(pca_dim,1)=1-testc(hangtr_set,hangtr_w);

hangte_set=prdataset(feature_test,label_test);
acc(pca_dim,2)=1-testc(hangte_set,hangtr_w);
end

predict_label=labeld(hangte_set,hangtr_w);
[hang_cf,~,~,~] = confmat(label_test,predict_label);
for i=1:5
hang_cf(i,:)=hang_cf(i,:)/sum(hang_cf(i,:));
end
hang_cf