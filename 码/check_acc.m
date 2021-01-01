function acc = check_acc(predict_label, label)
% the input should be the predict label and the real label
% the output is the accuracy of the pair, and is in the format of Decimal

[height, ~] = size(label);
[height1, ~] = size(predict_label);
if height1 == 1
    predict_label = predict_label';
    [height1, ~] = size(predict_label);
end
if height == 1
    label = label';
    [height, ~] = size(label);
end
if height~=height1
    disp('Please check the dimention of two label!')
end
count = 0;
for i=1:height
    if predict_label(i,1) ~= label(i,1)
        count = count + 1;
    end
end
acc = count/height;