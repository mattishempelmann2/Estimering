% TTT4275 - Classification of handwritten numbers.
% Task 1: Nearest-neighbour (NN) classifier using the whole training set as templates

clear; close all; clc;
load('data_all.mat');

%Cast to double so that distance calculations are done in floating point.
trainv = double(trainv);
testv  = double(testv);
trainlab = double(trainlab);
testlab  = double(testlab);

numTrain  = size(trainv,1);           
numTest   = size(testv,1);          
numClass  = 10;                       
chunkSize = 1000;                     

%Equation 8
trainNormSq = sum(trainv.^2, 2)';     

predLab = zeros(numTest,1);           
nnIdx   = zeros(numTest,1);        % index of the nearest training template

fprintf('Task 1, NN classifier with full training set (%d templates)\n', numTrain);
tic
for chunkStart = 1:chunkSize:numTest
    chunkEnd  = min(chunkStart + chunkSize - 1, numTest);
    X         = testv(chunkStart:chunkEnd, :);                  
    testNormSq = sum(X.^2, 2);                                  

    %Finding all distances
    D2 = bsxfun(@plus, testNormSq, trainNormSq) - 2*(X * trainv.'); 

    [~, idx] = min(D2, [], 2);      % nearest template per sample
    nnIdx(chunkStart:chunkEnd)   = idx;
    predLab(chunkStart:chunkEnd) = trainlab(idx);

    fprintf('   processed %5d / %5d test samples\n', chunkEnd, numTest);
end
runtime = toc;
fprintf('NN classification finished in %.1f s\n', runtime);

%Confusion matrix and error rate
confMat = zeros(numClass, numClass);
for i = 1:numTest
    confMat(testlab(i)+1, predLab(i)+1) = confMat(testlab(i)+1, predLab(i)+1) + 1;
end

errorRate = 1 - trace(confMat)/numTest;
fprintf('\nTask 1 - NN, all 60000 templates\n');
fprintf('Error rate : %.2f %%   (%d / %d misclassified)\n', ...
        100*errorRate, numTest - trace(confMat), numTest);
disp('Confusion matrix (rows = true class 0..9, columns = predicted 0..9):');
disp(confMat);


% Plot misclassified and correctly classified numbers
wrongIdx   = find(predLab ~= testlab);
correctIdx = find(predLab == testlab);

numShow = 8;                                      
rng(0);                                           
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
