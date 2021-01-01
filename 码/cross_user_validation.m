%%
clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
[feature_train, label_train, user, v, mstd] = preprocessing_svm(filename_train, 16, 7, 0, 0, 3, 0);
%[feature_test, label_test, user1, v, mstd] = preprocessing_svm(filename_test, 16, 7, 1, v, 3, mstd);
%%
user = user + 1;
[height,width]=size(feature_train);
flag=ones(12,1);
user_index=zeros(12,800);
for i=1:length(label_train)
    user_index(user(i,1),flag(user(i,1)))=i;
    flag(user(i,1))=flag(user(i,1))+1;
end
mod_index=user_index([1 2 3 6 7 9 10 11 12], 1:800);

acc_mean=zeros(120,60);
acc_std=zeros(60,60);
ftrain=zeros(8*800,width);
ltrain=zeros(8*800,1);
fvalid=zeros(800,width);
lvalid=zeros(800,1);

for c = 1:120
    for g = 1:60
        cost = 10^((c/10)-6);
        gamma = 10^((g/10)-3);
        libsvm_options=['-s 0  -c ',mat2str(c),' -g ',mat2str(g),' -t 2'];
        acc=zeros(9,1);
        for i=1:9
            for j=1:8
                ftrain((j-1)*800+1:j*800,:)=feature_train(mod_index(j,:),:);
                ltrain((j-1)*800+1:j*800,1)=label_train(mod_index(j,:),1);
            end
            fvalid=feature_train(mod_index(9,:),:);
            lvalid=label_train(mod_index(9,:),1);
            model = svmtrain(ltrain, ftrain, libsvm_options);
            [~, accuracy, ~] = svmpredict(lvalid, fvalid, model);
            acc(i,1)=accuracy(1,1);
            mod_index = circshift(mod_index,[-1 0]);
        end
        acc_mean(c,g)=mean(acc);
        acc_std(c,g)=std(acc);
    end
end
[M,I]=max(max(acc_mean))