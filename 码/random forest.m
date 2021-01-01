% clear
% pca_dimention=16;
% threshold=7;
% type=1;
clear
filename_train = 'D_train_large.csv';
filename_test = 'D_test.csv';
tree_num=170;
acc=zeros(1,2);
acc_mean=zeros(1,2);
pca_train=1;%change 1 if want pca
pca_test=2;%change 2 if want pca
type=13;
pca_dim=13;
thr=0;
V=0;%dont change
for tree_num=170
   tree_num
    for temp=1
        temp
        [feature_train, label_train, user,V] = preprocessing_simple(filename_train,thr,  ...
            pca_train,pca_dim,V,type);
        [feature_test, label_test, user1,V] = preprocessing_simple(filename_test,thr,  ...
            pca_test,pca_dim,V,type);
        %standardize
        feature_size=length(feature_train(1,:));
        train_number=length(feature_train(:,1));
        test_number=length(feature_test(:,1));
        mean=zeros(feature_size,1);
        sigma=zeros(feature_size,1);
        train_std=zeros(train_number,feature_size);
        test_std=zeros(test_number,feature_size);
        for t=1:feature_size
            [train_std(:,t),mean(t,1),sigma(t,1)]=zscore(feature_train(:,t));
            test_std(:,t)=(feature_test(:,t)-mean(t,1))/sigma(t,1);
        end
        feature_train=zscore(feature_train);
        feature_test=test_std;
        
        %normalization
        
        % large=max(feature_train);
        % small=min(feature_train);
        % feature_train=(feature_train-small)/(large-small);
        % feature_test=(feature_test-small)/(large-small);
        
        B=TreeBagger(tree_num,feature_train,label_train,'Method', 'classification');
        predict_test=predict(B,feature_test);
        predict_train=predict(B,feature_train);
        predict_test_label=str2num(cell2mat(predict_test));
        predict_train_label=str2num(cell2mat(predict_train));
        acc(temp,1)=length(find(predict_test_label == label_test))/length(label_test);
        acc(temp,2)=length(find(predict_train_label == label_train))/length(label_train);
    end
    acc_mean(tree_num/10,1)=sum(acc(:,1))/1;
    acc_mean(tree_num/10,2)=sum(acc(:,2))/1;
end

[hang_cf,~,~,~] = confmat(label_test,predict_test_label);
for i=1:5
hang_cf(i,:)=hang_cf(i,:)/sum(hang_cf(i,:));
end
hang_cf

