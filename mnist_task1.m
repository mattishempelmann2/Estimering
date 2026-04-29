%% =====================================================================
%  TTT4275 - Classification of handwritten numbers (MNIST)
%  Task 1: Nearest-neighbour (NN) classifier using the whole training
%          set as templates (Euclidean distance, chunk-based evaluation).
%
%  Expects the file data_all.mat (produced by read09.m) in the same
%  directory.  Variables in data_all.mat:
%       trainv   (60000 x 784, uint8)   training vectors
%       trainlab (60000 x 1,   uint8)   training labels (0..9)
%       testv    (10000 x 784, uint8)   test vectors
%       testlab  (10000 x 1,   uint8)   test labels (0..9)
%       row_size, col_size = 28, vec_size = 784
% =====================================================================

clear; close all; clc;
load('data_all.mat');                 % provides trainv, trainlab, testv, testlab

% Cast to double so that distance calculations are done in floating point.
trainv = double(trainv);
testv  = double(testv);
trainlab = double(trainlab);
testlab  = double(testlab);

numTrain  = size(trainv,1);           % 60000
numTest   = size(testv,1);            %  10000
numClass  = 10;                       % digits 0..9
chunkSize = 1000;                     % evaluate 1000 test samples at a time

%% --------------------------------------------------------------------
%  Pre-compute once:  ||t_k||^2  for every training template.
%  Then the squared Euclidean distance between every test chunk X and
%  all templates T is obtained with the identity
%      || x - t ||^2  =  ||x||^2 + ||t||^2 - 2 x^T t,
%  where only the (-2 X T^T) term has to be recomputed per chunk.
% --------------------------------------------------------------------
trainNormSq = sum(trainv.^2, 2)';     % 1 x numTrain  (row vector)

predLab = zeros(numTest,1);           % nearest-neighbour prediction for every test sample
nnIdx   = zeros(numTest,1);           % index of the nearest training template (for plotting)

fprintf('Task 1 - NN classifier with full training set (%d templates)\n', numTrain);
tic
for chunkStart = 1:chunkSize:numTest
    chunkEnd  = min(chunkStart + chunkSize - 1, numTest);
    X         = testv(chunkStart:chunkEnd, :);                  % (chunk x 784)
    testNormSq = sum(X.^2, 2);                                  % (chunk x 1)

    % Squared Euclidean distance between every test row and every template.
    D2 = bsxfun(@plus, testNormSq, trainNormSq) - 2*(X * trainv.');  % chunk x numTrain

    [~, idx] = min(D2, [], 2);                                  % nearest template per sample
    nnIdx(chunkStart:chunkEnd)   = idx;
    predLab(chunkStart:chunkEnd) = trainlab(idx);

    fprintf('   processed %5d / %5d test samples\n', chunkEnd, numTest);
end
runtime = toc;
fprintf('NN classification finished in %.1f s\n', runtime);

%% --------------------------------------------------------------------
%  Confusion matrix and error rate
% --------------------------------------------------------------------
confMat = zeros(numClass, numClass);
for i = 1:numTest
    confMat(testlab(i)+1, predLab(i)+1) = confMat(testlab(i)+1, predLab(i)+1) + 1;
end

errorRate = 1 - trace(confMat)/numTest;
fprintf('\n=== Task 1 - NN, all 60000 templates ===\n');
fprintf('Error rate : %.2f %%   (%d / %d misclassified)\n', ...
        100*errorRate, numTest - trace(confMat), numTest);
disp('Confusion matrix (rows = true class 0..9, cols = predicted 0..9):');
disp(confMat);

save('mnist_task1_result.mat', 'confMat', 'errorRate', 'predLab', 'nnIdx', 'runtime');

%% --------------------------------------------------------------------
%  Plot some misclassified and correctly classified images
% --------------------------------------------------------------------
wrongIdx   = find(predLab ~= testlab);
correctIdx = find(predLab == testlab);

numShow = 8;                                      % images per figure
rng(0);                                           % reproducible selection
wrongSel   = wrongIdx(  randperm(length(wrongIdx),   numShow));
correctSel = correctIdx(randperm(length(correctIdx), numShow));

figure('Name','Misclassified digits (NN, 60k templates)');
for k = 1:numShow
    subplot(2, numShow/2, k);
    x = zeros(28,28); x(:) = testv(wrongSel(k), :);
    image(x'); colormap(gray(256)); axis image off;
    title(sprintf('true %d / pred %d', testlab(wrongSel(k)), predLab(wrongSel(k))));
end
saveas(gcf,'mnist_task1_misclassified.png');

figure('Name','Correctly classified digits (NN, 60k templates)');
for k = 1:numShow
    subplot(2, numShow/2, k);
    x = zeros(28,28); x(:) = testv(correctSel(k), :);
    image(x'); colormap(gray(256)); axis image off;
    title(sprintf('true %d / pred %d', testlab(correctSel(k)), predLab(correctSel(k))));
end
saveas(gcf,'mnist_task1_correct.png');
