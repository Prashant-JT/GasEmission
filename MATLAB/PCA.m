clc; clear;

dataset = readtable("Datos/gt_2015.csv");
headers = dataset.Properties.VariableNames;
features = dataset.Variables;
features = centerData(features);

[V_EIG, PCA_EIG, D_EIG] = compute_PCA_EIG(features); % Se computa con autovalores y autovectores
[V_SVD, PCA_SVD, D_SVD] = compute_PCA_SVD(features); % Se computa con la codificación SVD
[V_FUNC, PCA_FUNC, D_FUNC, TSQUARED, P_FUNC] = pca(features); % Se computa con la función de fábrica

P_TOTAL = (D_EIG/sum(D_EIG)); % Se calcula el porcentaje que contribuye cada componente.
percentage = sum(P_TOTAL(1:3)); % Se representa el 92% de los datos representando 3 componentes.

fprintf("Se representa el %f %% de los datos (3 componentes principales)\n", percentage*100);

figure(1);
scatter3(PCA_EIG(:, 1), PCA_EIG(:, 2), PCA_EIG(:, 3));
title("PCA (3 primeras componentes)");
xlabel("X"); ylabel("Y"), zlabel("Z");

function [V_, PCA_, D_] = compute_PCA_EIG(X)
    Z = (X'*X)/(size(X,1)); 

    [V_, D_] = eig(Z);
    [D_, order] = sort(diag(D_), 'descend');  
    V_ = V_(:,order);
                
    PCA_ = X*V_;
end

function [V_, PCA_, D_] = compute_PCA_SVD(X)
    Z = (X'*X)/(size(X,1));  % C = cov(XC);
    
    [U, S, ~] = svd(Z); % Perform Singular Value Decomposition
    
    V_ = U; D_ = S; 
   
    D_ = diag(D_);
          
    PCA_ = X*V_; % Get reduced data 
end

function XC = centerData(X)
    meanFeatures = mean(X, 1);
    XC = zeros(size(X));
    for i=1:size(X, 2)
        XC(:, i) = X(:, i) - meanFeatures(i);
    end     
end

