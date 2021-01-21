clc; clear;

dataset = readtable("dataset/gt_2015.csv");
headers = dataset.Properties.VariableNames;
features = normalize(dataset.Variables);

[V, PCA, D] = computePCA(features);

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(features);

function [V_, PCA_, D_] = computePCA(X)
    % X = bsxfun(@minus, X, mean(X,1)); % Mucho Pro
    meanFeatures = mean(X, 1);
    XC = zeros(size(X));
    for i=1:size(X, 2)
        XC(:, i) = X(:, i) - meanFeatures(i);
    end   
    
    Z = (X'*X)/(size(X,1));  % C = cov(XC);

    [V_, D_] = eig(Z);
    [D_, order] = sort(diag(D_), 'descend');  
    V_ = V_(:,order);

    PCA_ = X*V_;
end
