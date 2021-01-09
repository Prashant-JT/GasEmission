clc; clear;

dataset = readtable("dataset/gt_2015.csv");
headers = dataset.Properties.VariableNames;
features = dataset.Variables;

meanFeatures = mean(features, 1);

XC = zeros(size(features));
for i=1:size(features, 2)
    XC(:, i) = features(:, i) - meanFeatures(i);
end

Z = (XC'*XC)/size(features, 1);

[V, D] = eig(Z);
diagonal = diag(D)';

%figure(1)
%scatter(features(:, 2), features(:, 3));
figure(1)
scatter(XC(:, 1), XC(:, 3));

figure(2)
mult = features * V;
scatter(mult(:, 1), mult(:, 3));

%{
scatter(V);

m = 5000; n = 2;
A = randn(m,n);
% deformación por un factor de 3
A(:,2) = 3*A(:,2);
% matriz de rotación
phi = 45;
cose = cosd(phi); sen = sind(phi);
R = [cose -sen; sen cose];
% rotación y traslacion al punto (10,10)
B = A*R + 10;
scatter(B(:, 1), B(:, 2));
%}
