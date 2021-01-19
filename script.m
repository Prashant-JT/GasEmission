clc; clear;

dataset = readtable("dataset/gt_2015.csv");
headers = dataset.Properties.VariableNames;
features = normalize(dataset.Variables);

meanFeatures = mean(features, 1);

XC = zeros(size(features));
for i=1:size(features, 2)
    XC(:, i) = features(:, i) - meanFeatures(i);
end

Z = (XC'*XC)/size(features, 1);

[V, D] = eig(Z);
diagonal = diag(D)';

% scatter((1:11), diagonal);

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(features);

figure(1);
scatter3(SCORE(:,1),SCORE(:,2),SCORE(:,3));


Y = SCORE*COEFF;
figure(2);
scatter3(Y(:,1),Y(:,2),Y(:,3));

%{
figure('Name','EigenVectors','NumberTitle','off');
bar(COEFF);

plotpca={'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11'};
for p=1:11
    figure('Name',plotpca{p},'NumberTitle','off');

    x = 1:size(SCORE, 1);
    scatter(x, SCORE(:, p), 'r')
    xlabel('Timestamp in order') 
    ylabel("Feature Values("+ plotpca{p}+")")    
end
%}

