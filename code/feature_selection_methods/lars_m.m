function [ betaLtst ] = lars_m(X, y, par)  
[n,p] = size(X);
muA = zeros(n,1);  % estimation of the dependent variable
beta = zeros(p,1); % estimation of parameters
betaLtst = beta';  % keep parameters in a storage
 
for i = 1:p
    % correlation coefficients between each feature (column of X) and vector of residuals
    c = X'*(y-muA); % note that columns of X are centered and normalized
    [C, A] = max(abs(c)); % find maximal value of correlation and corresponding index j of column in X
    % A = find(C == abs(c));
    % Aplus = find(C==c); % never used
    Sj  = sign(c(A)); % get sign of j-th correlation coefficient
    XA = X(:,A).*(ones(n,1)*Sj'); %
    % XA = X(:,A)*Sj; %
    G = XA'*XA; % norm of XA
    oA = ones(1,length(A)); % vector of ones
    AA =(oA*G^(-1)*oA')^(-0.5); % inverse matrix in the normal equation
 
    wA = AA*G^(-1)*oA'; % parameters to compute the unit bisector
    uA = XA*wA; % compute the unit bisector
    a = X'*uA; % product vector to compute new gamma
    if i<p % for all columns of X but the last
        M = [(C-c)./(AA-a);(C+c)./(AA+a)];
        M(find(M<=0)) = +Inf;
        gamma = min(M);
    else
        gamma  = C/AA;
    end
 
    muA = muA + gamma*uA; % make new approximation of the dependent variable
    beta(A) = beta(A) + gamma*wA.*Sj; % make new parameters
    betaLtst = [betaLtst; beta']; % store the parameters at k-th step
end
end
