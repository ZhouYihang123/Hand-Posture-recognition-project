function [feature, label, user,V] = preprocessingold(filename, threshold, pca,pca_dim,V,type)
% filename is the name of dataset file
% pca_dimention is the dimention you want to reserve after pca
% threshold means the least marker requirement
% if type = 1, then the other selected data will be used 
%% Read data
% the first column is label
% the second column is user
Data = csvread(filename, 1, 1);
raw_Data_label = Data(:,1);
raw_Data_user = Data(:,2);
raw_Data_3d = Data(:,3:1:35);
[raw_height, ~]=size(raw_Data_3d);
clearvars Train

%% data filter function
% feature1: 
flag=1;
count=0;
for i=1:raw_height
    for j=1:11
        if raw_Data_3d(i,(j-1)*3+1)~=0
            count=count+1;
        else
            break
        end
    end
    if count>=threshold
        Data_3d(flag,:) = raw_Data_3d(i,:);
        Data_label(flag,:) = raw_Data_label(i,:);
        Data_user(flag,:)= raw_Data_user(i,:);
        datanum(flag,1) = count; 
        flag=flag+1;
    end
    count=0;
end
% Train_3d is the feature that can be used
% Train_user and Train_label are the correspond data
%% extract the recomended feature
[height, ~]=size(Data_3d);
% height is the number of filtered data
if type==13
    feature = zeros(height, 13);
end
if type==14
    feature=zeros(height,13);
end
if type==16
    feature = zeros(height, 16);
end
if type==19
    feature = zeros(height, 19);
end
if type==22
    feature = zeros(height, 22);
end

feature(:,1) = datanum;     % feature1 = marker number

count=0;    % record the number of nonzero marker in each row
for i=1:height
    for j=1:11
        if Data_3d(i,(j-1)*3+1)~=0
            count=count+1;
        end
    end
    feature(i,2) = min(Data_3d(i,1:3:(count*3-2)));    % feature2 = x min
    feature(i,3) = min(Data_3d(i,2:3:(count*3-1)));    % feature3 = y min
    feature(i,4) = min(Data_3d(i,3:3:(count*3)));      % feature4 = z min
    feature(i,5) = std(Data_3d(i,1:3:(count*3-2)));     % feature5 = x std
    feature(i,6) = std(Data_3d(i,2:3:(count*3-1)));     % feature6 = x std
    feature(i,7) = std(Data_3d(i,3:3:(count*3)));       % feature7 = z std
    if type==13||type==16||type==19||type==22
            feature(i,8) = max(Data_3d(i,1:3:(count*3-2)));     % feature8 = x max
            feature(i,9) = max(Data_3d(i,2:3:(count*3-1)));     % feature9 = y max
            feature(i,10) = max(Data_3d(i,3:3:(count*3)));   	% feature10 = z max
    end
    if type==14
    feature(i,8) = max(abs(Data_3d(i,1:3:(count*3-2))));     % feature8 = x max
    feature(i,9) = max(abs(Data_3d(i,2:3:(count*3-1))));     % feature9 = y max
    feature(i,10) = max(abs(Data_3d(i,3:3:(count*3))));   	% feature10 = z max
    end
    feature(i,11) = mean(Data_3d(i,1:3:(count*3-2)));    % feature8 = x mean
    feature(i,12) = mean(Data_3d(i,2:3:(count*3-1)));    % feature9 = y mean
    feature(i,13) = mean(Data_3d(i,3:3:(count*3)));      % feature10 = z mean
    if type==16||type==19||type==22
    feature(i,14) = skewness(Data_3d(i,1:3:(count*3-2)));     % feature14 = x std
    feature(i,15) = skewness(Data_3d(i,2:3:(count*3-1)));     % feature15 = x std
    feature(i,16) = skewness(Data_3d(i,3:3:(count*3)));       % feature16 = z std
    end
    count=0;
end
%% transform to sphere coordinate
if type==19||type==22
    Train_sph=zeros(height, 33);
    % [a,b,r]=[alph, beta, radius]
    for i=1:height
        for j=1:11
            if Data_3d(i,(j-1)*3+1)~=0
                [Train_sph(i,(j-1)*3+1),Train_sph(i,(j-1)*3+2),Train_sph(i,j*3)]=...
                    cart2sph(Data_3d(i,(j-1)*3+1),Data_3d(i,(j-1)*3+2),Data_3d(i,j*3));
            end
        end
    end
    count=0;    % record the number of nonzero marker in each row
    sph_std = zeros(height,3);
    sph_mean = zeros(height,3);
    new_3d = zeros(height,3);
    for i=1:height
        for j=1:11
            if Train_sph(i,(j-1)*3+1)~=0
                count=count+1;
            end
        end
        sph_mean(i,1)=mean(Train_sph(i,1:3:(count*3-2)));
        sph_mean(i,2)=mean(Train_sph(i,2:3:(count*3-1)));
        sph_mean(i,3)=mean(Train_sph(i,3:3:(count*3)));
        Train_sph(i,1:3:(count*3-2))=Train_sph(i,1:3:(count*3-2))-sph_mean(i,1);
        Train_sph(i,2:3:(count*3-1))=Train_sph(i,2:3:(count*3-1))-sph_mean(i,2);
        Train_sph(i,3:3:(count*3))=Train_sph(i,3:3:(count*3))-sph_mean(i,3);
        sph_std(i,1)=std(Train_sph(i,1:3:(count*3-2)));
        sph_std(i,2)=std(Train_sph(i,2:3:(count*3-1)));
        sph_std(i,3)=std(Train_sph(i,3:3:(count*3))); 
        new_3d(i,:) = sph2cart(sph_mean(i,1),sph_mean(i,2),sph_mean(i,3));
        count=0;
    end
    feature(:,17:1:19) = sph_std;
    if type==22
    feature(:,20:1:22) = new_3d;
    end
end
    % feature14 = sphere coordinate std (a)
    % feature15 = sphere coordinate std (b)
    % feature16 = sphere coordinate std (r)
    % feature17 = sphere coordinate mean (a)
    % feature18 = sphere coordinate mean (b)
    % feature19 = sphere coordinate mean (r)
    %% PCA
    if pca ==1
        [~, ~, V] = svd(feature'*feature);
        feature = feature*V(:, 1:1:pca_dim);
    end
    if pca==2
        feature = feature*V(:, 1:1:pca_dim);
    end
    if pca==0
        V="None";
    end
label = Data_label;
user = Data_user;
%[feature,~,~] = zscore(feature);
end
 
