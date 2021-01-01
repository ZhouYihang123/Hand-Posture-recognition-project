function [mean_acc,std_acc] = crossval_svm(data,c,g)
% data: first collum is label, others are features
libsvm_options=['-s 0  -c ',mat2str(c),' -g ',mat2str(g),' -t 2'];
cv = cvpartition(data(:,1),'KFold',5,'Stratify',true);
acc=zeros(cv.NumTestSets,1);
for i=1:cv.NumTestSets
    idx=training(cv,i);
    alldata=[idx data];
    flag1=0;
    flag2=0;
    for j=1:89
        if idx(j,1)==1
            flag1=flag1+1;
            train_feature(flag1,1:2)=alldata(j,3:4);
            train_label(flag1,1)=alldata(j,2);
        else
            flag2=flag2+1;
            test_feature(flag2,1:2)=alldata(j,3:4);
            test_label(flag2,1)=alldata(j,2);
        end
    end
    model = svmtrain(train_label, train_feature, libsvm_options);
    [~, accuracy, ~] = svmpredict(test_label, test_feature, model);
    acc(i,1)=accuracy(1,1);
end

mean_acc=mean(acc);
std_acc=std(acc);
