clc; clear;

dataset = readtable("dataset/gt_2015.csv");
headers = dataset.Properties.VariableNames;
features = normalize(dataset.Variables);
features = centerData(features);

[V_EIG, PCA_EIG, D_EIG] = compute_PCA_EIG(features);
[V_SVD, PCA_SVD, D_SVD] = compute_PCA_SVD(features);
[V_FUNC, PCA_FUNC, D_FUNC, TSQUARED, P_FUNC] = pca(features);

P_TOTAL = (D_EIG/sum(D_EIG));
percentage = sum(P_TOTAL(1:3)); % Se representa el 83% de los datos

figure(1);
scatter3(PCA_EIG(:, 1), PCA_EIG(:, 2), PCA_EIG(:, 3));

function [V_, PCA_, D_] = compute_PCA_EIG(X)
    Z = (X'*X)/(size(X,1));  % C = cov(X);

    [V_, D_] = eig(Z);
    [D_, order] = sort(diag(D_), 'descend');  
    V_ = V_(:,order);
            
    PCA_ = X*V_;
end

function [V_, PCA_, D_] = compute_PCA_SVD(X)
    Z = (X'*X)/(size(X,1));  % C = cov(XC);
    
    [U, S, ~] = svd(Z); % Perform Singular Value Decomposition
    
    %{
    What are eigenvalues? What are singular values? 
    They both describe the behavior of a matrix on a 
    certain set of vectors. The difference is this: 
    The eigenvectors of a matrix describe the directions 
    of its invariant action. The singular vectors of a 
    matrix describe the directions of its maximum action. 
    And the corresponding eigen- and singular values describe 
    the magnitude of that action.
    %}
    V_ = U; D_ = S; 
   
    D_ = diag(D_);
          
    PCA_ = X*V_; % Get reduced data 
end

function XC = centerData(X)
    % X = bsxfun(@minus, X, mean(X,1)); % Mucho Pro
    meanFeatures = mean(X, 1);
    XC = zeros(size(X));
    for i=1:size(X, 2)
        XC(:, i) = X(:, i) - meanFeatures(i);
    end     
end
