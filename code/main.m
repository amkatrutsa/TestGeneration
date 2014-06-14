clear all
addpath('./criteria')
addpath('./feature_selection_methods')
rng(0);
objects = 1000;
features.rand_features = 0;
features.ortfeat_features = 0;
features.coltarget_features = 50;
features.colfeat_features = 0;
features.ortcol_features = 0;
param.multpar = 0.8;
param.target = randi(1.5 * objects, objects, 1);
param.iter = 5;
param.s_0 = 0.5;
alg = {'Lasso', 'LARS', 'Stepwise', 'ElasticNet', 'Ridge'};
crit = {'stability', 'Cp', 'RSS', 'CondNumber', 'Vif', ...
         'Rsq_adj', 'bic', 'ftest'};
% crit = {'ftest'};
% alg = {'Ridge'};
[ matAlgCrit, learn_error, test_error, num_param ] = AlgCrit(alg, crit, ...
                                                    objects, features, param);
% [ Vif, k ] = PlotVif_k(alg, objects, features, param);
% [ Stability, s0 ] = Plotd_s0(alg, objects, features, param);