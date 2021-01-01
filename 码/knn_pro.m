clear
filename_train = 'D_train.csv';
filename_test = 'D_test.csv';
% num = 6;
% [feature_train, label_train, user, v, mstd] = preprocessing_knn(filename_train, 16, 7, 0, 0, 3, 0);
% [feature_test, label_test, user1, v, mstd] = preprocessing_knn(filename_test, 16, 7, 1, v, 3, mstd);

num = 33;
[feature_train, label_train, user, v, mstd] = preprocessing_knn(filename_train, 22, 6, 0, 0, 2, 0);
[feature_test, label_test, user1, v, mstd] = preprocessing_knn(filename_test, 22, 6, 1, v, 2, mstd);
% [V,~] = eig(feature_train'*feature_train);
% % Conv=cov(feature_train);
% % [V,~]=eig(Conv);
% feature_train = feature_train*V;
% feature_test = feature_test*V;

count1=1;
count2=1;
count3=1;
count4=1;
count5=1;
for i=1:length(label_train)
    if label_train(i,1)==1
        fclass1(count1,:)=feature_train(i,:);
        count1=count1+1;
    end
    if label_train(i,1)==2
        fclass2(count2,:)=feature_train(i,:);
        count2=count2+1;
    end
    if label_train(i,1)==3
        fclass3(count3,:)=feature_train(i,:);
        count3=count3+1;
    end
    if label_train(i,1)==4
        fclass4(count4,:)=feature_train(i,:);
        count4=count4+1;
    end
    if label_train(i,1)==5
        fclass5(count5,:)=feature_train(i,:);
        count5=count5+1;
    end
end
[~,centroid1] = kmeans(fclass1,num);
[~,centroid2] = kmeans(fclass2,num);
[~,centroid3] = kmeans(fclass3,num);
[~,centroid4] = kmeans(fclass4,num);
[~,centroid5] = kmeans(fclass5,num);
centroid = [centroid1;centroid2;...
    centroid3;centroid4;centroid5];
% for i=1:25
%     centroid(i,:)=centroid(i,:)/sqrt(centroid(i,:)*centroid(i,:)');
% end
[height,~]=size(label_test);
predict_label = zeros(height,1);
% for i=1:length(label_test)
%     distance = sqrt((centroid-feature_test(i,:))*(centroid-feature_test(i,:))');
%     [~,I]=sort(distance);
%     predict_label(i,1) = mode(ceil(I(1:1:ceil(num/4))/num));
% end
% 
for i=1:length(label_test)
    distance=zeros(5,num);
    power=zeros(5,1);
    for k=1:5
        for j=1:num
            distance(k,j)=sqrt((centroid((k-1)*num+j,:)-feature_test(i,:))*(centroid((k-1)*num+j,:)-feature_test(i,:))');
        end
        [B,~]=sort(distance(k,:));
        power(k,1)=1/sum((B(1:1:floor(num/4))));
    end
    [~,predict_label(i,1)]=max(power);
end
acc=length(find(predict_label == label_test))/(length(label_test));