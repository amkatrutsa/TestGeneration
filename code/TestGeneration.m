% Script to create data set with definite structure of the features,
% compare quality measures from the folder criteria for 
% the obtained model from folder feature_selection_method, 
% get dependence VIF on the multocollinerity parameter and 
% get dependence number of the redundant features from the limit error.
%
% Copyright Alexandr Katrutsa (c) 05-06.2014

clear all
addpath('./criteria')
addpath('./feature_selection_methods')
% rng(0);
% Parameters for data sets generation
numObjects = 1000;
features.rand_features = 0;
features.ortfeat_features = 10;
features.coltarget_features = 0;
features.colfeat_features = 40;
features.ortcol_features = 0;
param.multpar = 0.7; % multicollinearity parameter
param.target = randi(1.5 * numObjects, numObjects, 1);
param.iter = 5;
param.s_0 = 0.5; % limit error
% Considered feature selection methods
alg = {'Lasso', 'LARS', 'Stepwise', 'ElasticNet', 'Ridge'};
% Considered quality measures
crit = {'stability', 'Cp', 'RSS', 'CondNumber', 'Vif', ...
         'Rsq_adj', 'bic', 'ftest'};
% Function returns the values of every quality measures for every FSM
% [ matAlgCrit, learn_error, test_error, num_param ] = AlgCrit(alg, crit, ...
%                                                     numObjects, features, param);
% [ TestError_logcond, TestError_numpar ] = AlgStable(alg, crit, numObjects, features, param);
% Function returns the pair VIF and corresponding k (multicollinearity parameter)
% [ Vif, k ] = PlotVif_k(alg, numObjects, features, param);
% Function returns the number of the redundant features and 
% corresponding limit error 
[ Stability, s0 ] = Plotd_s0(alg, numObjects, features, param);