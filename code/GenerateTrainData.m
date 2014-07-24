function [ X, y ] = GenerateTrainData( criterion, numSets, numObj, numFeat, ...
                                        par, alg)
% Function generates data set and target vector to learn the classification
% algorithm to forecast the best FSM using the corresponding data set 
% 
% Input:
% criterion - string - name of the external criterion which use 
% to find the best FSM for data set
% numSets - [1, 1] - a number of the sets every type (total number of types is 3)
% numObj - [1, 1] - number of the rows in generated data sets
% numFeat - [1, 1] - number of the columns in generated data sets
% par - structure with additional parameters:
%       multpar - [1, 1] - multicollinearity parameter
%       s_0 - [1, 1] - a limit error
% alg - {cell array} - a list of the considered FSM
%
% Output:
% X - [3 * numSets, numObj * numFeat] - learn data set
% y - [3 * numSets, 1] - vector with indices of the best FSM 
%                       for every generated data set
%
% Copyright Alexandr Katrutsa (c) 07.2014

X = zeros((3 * numSets), numObj * numFeat);
y = zeros((3 * numSets), 1);

features.rand_features = 0;
features.ortfeat_features = 0;
features.coltarget_features = 0;
features.colfeat_features = 0;
features.ortcol_features = 0;

for i = 1: (3 * numSets)
    par.target = randi(1.5 * numObj, numObj, 1);
    learn_y = par.target;
    switch mod(i, 3)
        case 0
            features.ortcol_features = numFeat;
        case 1
            features.coltarget_features = numFeat;
        case 2
            features.ortfeat_features = 0.2 * numFeat;
            features.colfeat_features = 0.8 * numFeat; 
    end
    D = CreateData(numObj, features, par);
%     Normalisation
    len = sum(D.^2).^0.5;
    D = D ./ repmat(len, size(D, 1), 1);
    learn_y = learn_y ./ norm(learn_y);
%     Vectorisation
    X(i, :) = VectorizedData(D);
    y(i) = IdxBestFSM(D, learn_y, criterion, alg, par);
    features.ortcol_features = 0;
    features.coltarget_features = 0;
    features.ortfeat_features = 0;
    features.colfeat_features = 0;
end
end