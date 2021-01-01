clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
threshold=0;
pca_train=1;
pca_test=2;
type=13;
V=0;%dont change
acc=zeros(1,2);
pca_dim=10;
[feature_train, label_train, user,V] = preprocessing_simple(filename_train,threshold,  ...
    pca_train,pca_dim,V,type);
[feature_test, label_test, user1,V] = preprocessing_simple(filename_test,threshold,  ...
    pca_test,pca_dim,V,type);

%std
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

save('feature.mat','feature_train','feature_test','label_train','label_test')
