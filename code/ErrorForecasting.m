clear all
addpath('./criteria')
addpath('./feature_selection_methods')
% rng(0);
% Number of the sets every type, total rows in X equals 3 * numDataSets
numDataSets = 10; 
numObj = 1000;
numFeat = 50;
param.multpar = 0.8;
param.s_0 = 1;
alg = {'Lasso', 'LARS', 'Stepwise', 'ElasticNet', 'Ridge'};
criter = 'stability';
% Generate train data set
fprintf('Start training!\n')
[ X, y ] = GenerateTrainData(criter, numDataSets, numObj, numFeat, param, alg);
% Train classifier
classTree = ClassificationTree.fit(X, y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numObj = 500;
numFeat = 30;
numDataSets.AdeqRedund = 20;
numDataSets.AdeqCorrel = 10;
numDataSets.InadeqCorrel = 50;
numDataSets.AdeqRandom = 0;
par.multpar = 0.6;
par.s_0 = 1;
[~, par.numTrainFeat] = size(X);
fprintf('Start testing!\n')
[testX, testY] = GenerateTestData(criter, numObj, numFeat, numDataSets, par, alg);
testLabel = predict(classTree, testX);
accuracy = sum(testLabel == testY) / length(testY)