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
thr=6;
V=0;%dont change
acc=zeros(1,1);
for pca_dim=13
    [feature_train, label_train, user,V] = preprocessing_simple(filename_train,thr,  ...
        pca_train,pca_dim,V,type);
    [feature_test, label_test, user1,V] = preprocessing_simple(filename_test,thr,  ...
        pca_test,pca_dim,V,type);
    %standardize
    feature_size=length(feature_train(1,:));
    train_number=length(feature_train(:,1));
    test_number=length(feature_test(:,1));
    mean1=zeros(feature_size,1);
    sigma1=zeros(feature_size,1);
    mean2=zeros(feature_size,1);
    sigma2=zeros(feature_size,1);
    train_std=zeros(train_number,feature_size);
    test_std=zeros(test_number,feature_size);
    for t=1:feature_size
        [train_std(:,t),mean1(t,1),sigma1(t,1)]=zscore(feature_train(:,t));
    end
    for k=150
        k
        md = fitcknn(feature_train,label_train,'NumNeighbors',k,'Standardize',1);
        predict_test_label=predict(md,feature_test);
        predict_train_label=predict(md,feature_train);
        acc(k/10,1)=length(find(predict_test_label == label_test))/length(label_test);
        acc(k/10,2)=length(find(predict_train_label == label_train))/length(label_train);
    end
end

[hang_cf,~,~,~] = confmat(label_test,predict_test_label);
for i=1:5
hang_cf(i,:)=hang_cf(i,:)/sum(hang_cf(i,:));
end
hang_cf