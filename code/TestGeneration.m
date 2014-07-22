clear all
addpath('./criteria')
addpath('./feature_selection_methods')
% rng(0);
numObjects = 1000;
features.rand_features = 0;
features.ortfeat_features = 10;
features.coltarget_features = 0;
features.colfeat_features = 40;
features.ortcol_features = 0;
param.multpar = 0.7;
param.target = randi(1.5 * numObjects, numObjects, 1);
param.iter = 5;
param.s_0 = 0.5;
alg = {'Lasso', 'LARS', 'Stepwise', 'ElasticNet', 'Ridge'};
crit = {'stability', 'Cp', 'RSS', 'CondNumber', 'Vif', ...
         'Rsq_adj', 'bic', 'ftest'};
% crit = {'ftest'};
% alg = {'Ridge'};
% [ matAlgCrit, learn_error, test_error, num_param ] = AlgCrit(alg, crit, ...
%                                                     numObjects, features, param);
% [ TestError_logcond, TestError_numpar ] = AlgStable(alg, crit, numObjects, features, param);
% [ Vif, k ] = PlotVif_k(alg, numObjects, features, param);
[ Stability, s0 ] = Plotd_s0(alg, numObjects, features, param);