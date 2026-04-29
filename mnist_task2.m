%  TTT4275 - Classification of handwritten numbers (MNIST)
%  Task 2: Cluster the training set (k-means, M = 64 per class) and
%          classify with (i) 1-NN and (ii) K-NN, K = 7, on the 640
%          cluster templates.
%
%  Expects data_all.mat (produced by read09.m) in the same directory.

clear; close all; clc;
load('data_all.mat');                 % provides trainv, trainlab, testv, testlab

trainv   = double(trainv);
testv    = double(testv);
trainlab = double(trainlab);
testlab  = double(testlab);

numTest  = size(testv,1);             % 10000
numClass = 10;                        % digits 0..9
M        = 64;                        % clusters per class
K        = 7;                         % K for K-NN

%  Per-class k-means clustering
%  For every class i the ~6000 training vectors are clustered into M
%  templates.  The 64*10 = 640 cluster centres are stacked into a
%  single template matrix C with a matching label vector Clab.
fprintf('Task 2, k-means clustering (M = %d per class)\n', M);
C    = zeros(numClass*M, 784);
Clab = zeros(numClass*M, 1);
tic
for i = 0:numClass-1
    trainvi = trainv(trainlab == i, :);              % training vectors for class i
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

%  (b) Nearest-neighbour classifier using the 640 cluster templates.
%      Because the template set is small we can compute the full
%      10000 x 640 distance matrix in one shot.
tic
CnormSq   = sum(C.^2, 2).';                         % 1 x 640
TestNormSq = sum(testv.^2, 2);                      % 10000 x 1
D2 = bsxfun(@plus, TestNormSq, CnormSq) - 2*(testv * C.');   % 10000 x 640

[~, nnIdx]  = min(D2, [], 2);
predLab_NN  = Clab(nnIdx);
nnTime      = toc;

confMat_NN = zeros(numClass, numClass);
for i = 1:numTest
    confMat_NN(testlab(i)+1, predLab_NN(i)+1) = confMat_NN(testlab(i)+1, predLab_NN(i)+1) + 1;
end
err_NN = 1 - trace(confMat_NN)/numTest;

fprintf('\n=== NN on %d cluster templates ===\n', numClass*M);
fprintf('Error rate : %.2f %%   (classification time %.2f s)\n', 100*err_NN, nnTime);
disp('Confusion matrix (rows = true, cols = predicted):');
disp(confMat_NN);

%  (c) K-NN classifier (K = 7) using the same 640 cluster templates.
%      For every test sample we pick the K nearest templates and vote
%      on the most frequent label.
tic
[~, sortedIdx] = sort(D2, 2, 'ascend');             % 10000 x 640
knnIdx = sortedIdx(:, 1:K);                         % 10000 x K
knnLab = Clab(knnIdx);                              % 10000 x K  (labels of K nearest)

predLab_KNN = zeros(numTest,1);
for i = 1:numTest
    predLab_KNN(i) = mode(knnLab(i,:));             % majority vote
end
knnTime = toc;

confMat_KNN = zeros(numClass, numClass);
for i = 1:numTest
    confMat_KNN(testlab(i)+1, predLab_KNN(i)+1) = confMat_KNN(testlab(i)+1, predLab_KNN(i)+1) + 1;
end
err_KNN = 1 - trace(confMat_KNN)/numTest;

fprintf('\n=== K-NN on %d cluster templates, K = %d ===\n', numClass*M, K);
fprintf('Error rate : %.2f %%   (classification time %.2f s)\n', 100*err_KNN, knnTime);
disp('Confusion matrix (rows = true, cols = predicted):');
disp(confMat_KNN);

save('mnist_task2_result.mat', ...
     'C','Clab', ...
     'confMat_NN','err_NN','predLab_NN','nnTime', ...
     'confMat_KNN','err_KNN','predLab_KNN','knnTime', ...
     'clusterTime','M','K');
