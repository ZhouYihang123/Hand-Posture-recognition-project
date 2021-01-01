clear
clc
%% Read data
% the first column is label
% the second column is user
Train = csvread('D_train.csv', 1, 1);
Test = csvread('D_test.csv', 1, 1);
raw_Train_label = Train(:,1);
raw_Train_user = Train(:,2);
raw_Train_3d = Train(:,3:1:38);
Test_label = Test(:,1);
Test_user = Test(:,2);
Test_3d = Test(:,3:1:38);
clearvars Train Test

%% data filter function
% feature1: 
flag=1;
count=0;
for i=1:13500
    for j=1:11
        if raw_Train_3d(i,(j-1)*3+1)~=0
            count=count+1;
        else
            break
        end
    end
    if count>=8
        Train_3d(flag,:)=raw_Train_3d(i,:);
        Train_label(flag,:)=raw_Train_label(i,:);
        Train_user(flag,:)=raw_Train_user(i,:);
        flag=flag+1;
    end
    count=0;
end
% Train_3d is the feature that can be used
% Train_user and Train_label are the correspond data

%% transform to sphere coordinate
[height, width]=size(Train_3d);
Train_sph=zeros(height, 33);
% [a,b,r]=[alph, beta, radius]
for i=1:height
    for j=1:11
        if Train_3d(i,(j-1)*3+1)~=0
            [Train_sph(i,(j-1)*3+1),Train_sph(i,(j-1)*3+2),Train_sph(i,j*3)]=...
                cart2sph(Train_3d(i,(j-1)*3+1),Train_3d(i,(j-1)*3+2),Train_3d(i,j*3));
        end
    end
end
%% standarlize every data and get the std of each data on (a,b,r)
% count the mean of a,b,r and remove the mean
count=0;    % record the number of nonzero marker in each row
sph_std=zeros(height,3);
for i=1:height
    for j=1:11
        if Train_sph(i,(j-1)*3+1)~=0
            count=count+1;
        end
    end
    Train_sph(i,1:3:(count*3-2))=Train_sph(i,1:3:(count*3-2))-(sum(Train_sph(i,1:3:(count*3-2)))/count);
    Train_sph(i,2:3:(count*3-1))=Train_sph(i,2:3:(count*3-1))-(sum(Train_sph(i,2:3:(count*3-1)))/count);
    Train_sph(i,3:3:(count*3))=Train_sph(i,3:3:(count*3))-(sum(Train_sph(i,3:3:(count*3)))/count);
    sph_std(i,1)=std(Train_sph(i,1:3:(count*3-2)));
    sph_std(i,2)=std(Train_sph(i,2:3:(count*3-1)));
    sph_std(i,3)=std(Train_sph(i,3:3:(count*3)));
    count=0;
end
% [x,y,z]=sph2cart(Train_sph(1,1:3:(10*3-2)),Train_sph(1,2:3:(10*3-1)),Train_sph(1,3:3:(10*3)));
% scatter3(x,y,z)
%% try plot
map = [ 1 0 0
    0 1 0
    0 0 1
    1 1 0
    1 0 1];
colormap(map)
figure(1);
count=1;
p=2; %the user we focus on
for i=1:height
    if Train_user(i,1)==p
        user(count,1)=i;
        count=count+1;
    end

end
scatter3(10*sph_std(user(:,1),1),10*sph_std(user(:,1),2),10*sph_std(user(:,1),3),5,Train_label(user(:,1),1))
count = 0;
for i=1:length(Train_label)
    if Train_label(i,1)==1
        count=count+1;
    end
end