function [ X, y ] = GenerateTestData( criterion, numObj, numFeat, ...
                                    numDataSets, par, alg)
% Function creates data set X and corresponding target vector y to test
% trained classifier
%
% Input:
% criterion - string - name of the considered criterion
% numObj - [1, 1] - number of the rows in every type generated data sets
% numFeat - [1, 1] - number of the features in every type generated data sets
% numDataSets - structure with following fields:
%               AdeqRedund - [1, 1] - number of the adequate redundant data
%               sets
%               AdeqCorrel - [1, 1] - number of the adequate correlated
%               data sets
%               InadeqCorrel - [1, 1] - number of the inadequate correlated
%               data sets
%               AdeqRandom - [1, 1] - number of the adequate random data
%               sets
% par - structure with following fields:
%       multpar - [1, 1] - multicollinearity parameter
%       s_0 - [1, 1] - limit error
%       numTrainFeat - [1, 1] - number of the features in training set
% alg - {cell array} - a list of the considered FSM
%
% Output:
% X - [totalDataSets, min(par.numTrainFeat, numObj * numFeat)] - test data set 
% y - [totalDataSets, 1] - corresponding target vector with FSM indices
%
% Copyright Alexandr Katrutsa (c) 07.2014

totalDataSets = numDataSets.AdeqRedund + numDataSets.AdeqCorrel + ...
                numDataSets.InadeqCorrel + numDataSets.AdeqRandom;
X = zeros(totalDataSets, par.numTrainFeat);
y = zeros(totalDataSets, 1);

idxAdeqRedund = 1:numDataSets.AdeqRedund;
idxAdeqCorrel = (max(idxAdeqRedund) + 1):(max(idxAdeqRedund) + 1 + numDataSets.AdeqCorrel);
idxInadeqCorrel = (max(idxAdeqCorrel) + 1):(max(idxAdeqCorrel) + 1 + numDataSets.InadeqCorrel);
idxAdeqRandom = (max(idxInadeqCorrel) + 1):(max(idxInadeqCorrel) + 1 + numDataSets.AdeqRandom);

features.rand_features = 0;
features.ortfeat_features = 0;
features.coltarget_features = 0;
features.colfeat_features = 0;
features.ortcol_features = 0;

for i = 1:totalDataSets
    par.target = randi(1.5 * numObj, numObj, 1);
    learn_y = par.target;
    if (ismember(i, idxAdeqRedund))
         features.coltarget_features = numFeat;
    end
    if (ismember(i, idxAdeqCorrel))
         features.ortfeat_features = 0.2 * numFeat;
         features.colfeat_features = 0.8 * numFeat;        
    end
    if (ismember(i, idxInadeqCorrel))
         features.ortcol_features = numFeat;
    end
    if (ismember(i, idxAdeqRandom))
       features.rand_features = numFeat; 
    end
    D = CreateData(numObj, features, par);
%   Normalisation
    len = sum(D.^2).^0.5;
    D = D ./ repmat(len, size(D, 1), 1);
    learn_y = learn_y ./ norm(learn_y);
%   Vectorisation
    vectorizeD = VectorizedData(D);
    X(i, 1:min(numObj * numFeat, par.numTrainFeat)) = ...
            vectorizeD(1:min(numObj * numFeat, par.numTrainFeat));
    y(i) = IdxBestFSM(D, learn_y, criterion, alg, par);
    features.ortcol_features = 0;
    features.coltarget_features = 0;
    features.ortfeat_features = 0;
    features.colfeat_features = 0;
    features.rand_features = 0;
end
end