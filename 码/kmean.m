clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
[feature_train, label_train, user, V] = preprocessing(filename_train, 13, 7, 0, 0, 1);
[feature_test, label_test, user1, v] = preprocessing(filename_test, 13, 7, 1, V, 1);

temp=1;
[~,c] = kmeans(feature_train,5);
idx=zeros(length(feature_test),1);
d=zeros(5,1);
for i=1:length(feature_test)
    for t=1:5
        d(t,1)=norm(feature_test(i,:)-c(t,:));
    end
    [~,y]=min(d);
    idx(i,1)=y;
end

new_idx=zeros(length(feature_test),1);

for t=1:5
    class=zeros(1,1);
    temp=1;
    for i=1:length(feature_test)
        if idx(i,1)==t
            class(temp,1)=label_test(i,1);
            temp=temp+1;
        end
    end
    label=mode(class);
    for i=1:length(feature_test)
        if idx(i,1)==t
            new_idx(i,1)=label;
        end
    end
end
acc=length(find(new_idx == label_test))/(length(feature_test));


    
    