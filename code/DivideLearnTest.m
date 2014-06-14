function [ learn_X, test_X, learn_y, test_y ] = DivideLearnTest( X, y )
% Function divides data set X and target vector y on the learning and 
% test data sets and normalise both sets and target vectors
%
% Input:
% X - [m, n] - given data set
% y - [m, 1] - given target vector
%
% Output:
% learn_X - [m / 2, n] - learn data set
% test_X - [m / 2, n] - test data set
% learn_y - [m / 2, 1] - learning part of the target vector
% test_y - [m / 2, 1] - test part of the target vector

learn_size = floor(size(y, 1) / 2);
learn_X = X(1:learn_size, :);
test_X = X((learn_size + 1):end, :);
learn_y = y(1:learn_size, :);
test_y = y((learn_size + 1):end, :);

len = sum(learn_X.^2).^0.5;
learn_X = learn_X ./ repmat(len, size(learn_X, 1), 1);
learn_y = learn_y ./ norm(learn_y);

len = sum(test_X.^2).^0.5;
test_X = test_X ./ repmat(len, size(test_X, 1), 1);
test_y = test_y ./ norm(test_y);
end

