% TTT4275 - Classification of handwritten numbers (MNIST)
% Task 2: Cluster the training set (M = 64 per class) and
% classify with (1.) 1-NN and (2.) 7-NN, on the 640 cluster templates.

clear; close all; clc;
load('data_all.mat');                 

trainv   = double(trainv);
testv    = double(testv);
trainlab = double(trainlab);
testlab  = double(testlab);

numTest  = size(testv,1);             
numClass = 10;                        
M        = 64;                        
K        = 7;                         

% Per-class k-means clustering

fprintf('Task 2, k-means clustering (M = %d per class)\n', M);
C    = zeros(numClass*M, 784);
Clab = zeros(numClass*M, 1);
tic
for i = 0:numClass-1
    trainvi = trainv(trainlab == i, :);              
    [~, Ci] = kmeans(trainvi, M, ...
                     'MaxIter',   100, ...
                     'Replicates',  1, ...
                     'Display',  'off');
    C(   i*M + (1:M), :) = Ci;
    Clab(i*M + (1:M))    = i;
    fprintf('   class %d : %5d samples -> %d clusters\n', i, size(trainvi,1), M);
end
clusterTime = toc;
fprintf('Clustering finished in %.1f s\n', clusterTime);

% (b) Nearest-neighbour classifier
tic
CnormSq   = sum(C.^2, 2).';                         
TestNormSq = sum(testv.^2, 2);                      
D2 = bsxfun(@plus, TestNormSq, CnormSq) - 2*(testv * C.');   % 10000 x 640

[~, nnIdx]  = min(D2, [], 2);
predLab_NN  = Clab(nnIdx);
nnTime      = toc;

confMat_NN = zeros(numClass, numClass);
for i = 1:numTest
    confMat_NN(testlab(i)+1, predLab_NN(i)+1) = confMat_NN(testlab(i)+1, predLab_NN(i)+1) + 1;
end
err_NN = 1 - trace(confMat_NN)/numTest;

fprintf('\nNN on %d cluster templates\n', numClass*M);
fprintf('Error rate : %.2f %%   (classification time %.2f s)\n', 100*err_NN, nnTime);
disp('Confusion matrix (rows = true, columns = predicted):');
disp(confMat_NN);

% (c) K-NN classifier (K = 7) using the same 640 cluster templates.
% For every test sample we pick the K nearest templates and vote
% on the most frequent label.
tic
[~, sortedIdx] = sort(D2, 2, 'ascend');             
knnIdx = sortedIdx(:, 1:K);                         
knnLab = Clab(knnIdx);                              

predLab_KNN = zeros(numTest,1);
for i = 1:numTest
    predLab_KNN(i) = mode(knnLab(i,:));  % mode = majority vote
end
knnTime = toc;

confMat_KNN = zeros(numClass, numClass);
for i = 1:numTest
    confMat_KNN(testlab(i)+1, predLab_KNN(i)+1) = confMat_KNN(testlab(i)+1, predLab_KNN(i)+1) + 1;
end
err_KNN = 1 - trace(confMat_KNN)/numTest;

fprintf('\n7-NN on %d cluster templates.\n', K);
fprintf('Error rate : %.2f %%   (classification time %.2f s)\n', 100*err_KNN, knnTime);
disp('Confusion matrix (rows = true, columns = predicted):');
disp(confMat_KNN);

save('mnist_task2_result.mat', ...
     'C','Clab', ...
     'confMat_NN','err_NN','predLab_NN','nnTime', ...
     'confMat_KNN','err_KNN','predLab_KNN','knnTime', ...
     'clusterTime','M','K');
