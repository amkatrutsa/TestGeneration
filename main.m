clear all
addpath('./criteria')
addpath('./alg')
rng(0);
objects = 1000;
features.rand_features = 0;
features.ortfeat_features = 10;
features.coltarget_features = 0;
features.colfeat_features = 40;
features.ortcol_features = 0;
param.multpar = 0.8;
param.target = randi(1.5 * objects, objects, 1);
alg = {'Lasso', 'LARS', 'Stepwise', 'ElasticNet', 'Ridge'};
crit = {'stability', 'Cp', 'RSS', 'CondNumber', 'Vif', ...
         'Rsq_adj', 'bic', 'ftest'};
%crit = {'ftest'};
%alg = {'Ridge'};
X = CreateData(objects, features, param);
y = param.target;
len = sum(X.^2).^0.5;
X = X./repmat(len, size(X, 1), 1);
y = y ./ norm(y);
% learn_size = floor(size(y, 1) / 2);
% learn_X = X(1:learn_size, :);
% test_X = X((learn_size + 1):end, :);
% learn_y = y(1:learn_size, :);
% test_y = y((learn_size + 1):end, :);
param.iter = 5;
param.s_0 = 0.5;
matAlgCrit = AlgCrit(alg, crit, objects, features, param);
%[Vif, k] = PlotVif_k(alg, objects, features, param);
%[ Stability, s0 ] = Plotd_s0(alg, objects, features, param);
