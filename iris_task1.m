%% TTT4275 - The Iris Task - Part 1: Design/Training and Generalization
% Linear MSE classifier, implemented as
% described in the compendium part 3, chapters 2.4 and 3.2.
% eq. numbers reference the compendium

clear; close all; clc;

% Load dataset
% Each file: 50 samples x 4 features
x1all = load('class_1', '-ascii');   % Setosa     (50 x 4)
x2all = load('class_2', '-ascii');   % Versicolor (50 x 4)
x3all = load('class_3', '-ascii');   % Virginica  (50 x 4)

C  = 3;                              % number of classes
D  = size(x1all, 2);                 % feature dimension (4)
Ni = 50;                             % samples per class

%Parameters (tuned by trial and error, as specified in compendium p.17)
alpha    = 0.005;                    % step factor in Eq. 23
max_iter = 3000;                     % max gradient iterations

%  CASE 1: first 30 training, last 20 test
fprintf('=========================================================\n');
fprintf('CASE 1 : first 30 train, last 20 test\n');
fprintf('=========================================================\n');

[W1, mse1, conf_tr1, err_tr1, conf_te1, err_te1] = ...
    trainLinearMSE(x1all, x2all, x3all, 1:30, 31:50, C, alpha, max_iter);

fprintf('Final training MSE : %.6f\n\n', mse1(end));
fprintf('Training confusion matrix:\n'); disp(conf_tr1);
fprintf('Training error rate : %.2f %%\n\n', err_tr1);
fprintf('Test confusion matrix:\n');     disp(conf_te1);
fprintf('Test error rate     : %.2f %%\n', err_te1);

figure('Name', 'Case 1 - MSE Convergence');
plot(1:max_iter, mse1, 'b-', 'LineWidth', 1.3); grid on;
xlabel('Iteration m'); ylabel('MSE (Eq. 19)');
title('Case 1 : MSE convergence');
saveas(gcf, 'mse_case1.png');

%  CASE 2: last 30 training, first 20 test
fprintf('\n=========================================================\n');
fprintf('CASE 2 : last 30 train, first 20 test\n');
fprintf('=========================================================\n');

[W2, mse2, conf_tr2, err_tr2, conf_te2, err_te2] = ...
    trainLinearMSE(x1all, x2all, x3all, 21:50, 1:20, C, alpha, max_iter);

fprintf('Final training MSE : %.6f\n\n', mse2(end));
fprintf('Training confusion matrix:\n'); disp(conf_tr2);
fprintf('Training error rate : %.2f %%\n\n', err_tr2);
fprintf('Test confusion matrix:\n');     disp(conf_te2);
fprintf('Test error rate     : %.2f %%\n', err_te2);

figure('Name', 'Case 2 - MSE Convergence');
plot(1:max_iter, mse2, 'r-', 'LineWidth', 1.3); grid on;
xlabel('Iteration m'); ylabel('MSE (Eq. 19)');
title('Case 2 : MSE convergence');
saveas(gcf, 'mse_case2.png');

%Summary
fprintf('\n=========================================================\n');
fprintf('SUMMARY\n');
fprintf('=========================================================\n');
fprintf('                       Train err   Test err\n');
fprintf('Case 1 (first 30 tr) : %6.2f %%   %6.2f %%\n', err_tr1, err_te1);
fprintf('Case 2 (last 30 tr)  : %6.2f %%   %6.2f %%\n', err_tr2, err_te2);

%Helper: train and evaluate linear MSE classifier based on eq 19-23
%Reused for multiple tasks
function [W, mse_hist, conf_tr, err_tr, conf_te, err_te] = ...
    trainLinearMSE(x1all, x2all, x3all, trIdx, teIdx, C, alpha, max_iter)

    D = size(x1all, 2);

    %Split per class
    X_tr = [x1all(trIdx,:); x2all(trIdx,:); x3all(trIdx,:)];
    X_te = [x1all(teIdx,:); x2all(teIdx,:); x3all(teIdx,:)];
    Ntr  = size(X_tr, 1);    Nte = size(X_te, 1);
    Ntri = length(trIdx);    Ntei = length(teIdx);

    %Class labels (true classes, used for error/confusion)
    y_tr = [ones(Ntri,1); 2*ones(Ntri,1); 3*ones(Ntri,1)];
    y_te = [ones(Ntei,1); 2*ones(Ntei,1); 3*ones(Ntei,1)];

    % One-hot targets  t_k (Eq. 19)
    T_tr = zeros(Ntr, C);
    for k = 1:Ntr, T_tr(k, y_tr(k)) = 1; 
    end

    %add bias term ([x^T 1]^T, compendium p.15)
    Xa_tr = [X_tr, ones(Ntr, 1)];     % Ntr x (D+1)
    Xa_te = [X_te, ones(Nte, 1)];

    %Weight matrix W: (D+1) x C
    rng(7068);
    W = 0.01 * randn(D+1, C); %fill W with small random numbers. 
    % Could also be 0, works the same as long as not large value (convergence
    % stops for W = 10 * randn(D+1,C))

    mse_hist = zeros(max_iter, 1);

    %Gradient descent loop (batch)
    for m = 1:max_iter
        Z = Xa_tr * W;                            % Ntr x C   z_k = W*x_k
        G = 1 ./ (1 + exp(-Z));                   % Eq. 20, sigmoid

        E = G - T_tr;                             % (g_k - t_k)

        %Eq. 19
        mse_hist(m) = 0.5 * sum(E(:).^2);

        %Eq. 22
        grad_W = Xa_tr' * ( E .* G .* (1 - G) );  % (D+1) x C

        %eq. 23
        W = W - alpha * grad_W;
    end

    %Classification: argmax of discriminant g_i(x)
    [~, pred_tr] = max(Xa_tr * W, [], 2);
    [~, pred_te] = max(Xa_te * W, [], 2);

    %Confusion matrixes
    conf_tr = zeros(C, C);
    conf_te = zeros(C, C);
    for i = 1:C
        for j = 1:C
            conf_tr(i,j) = sum(pred_tr(y_tr == i) == j);
            conf_te(i,j) = sum(pred_te(y_te == i) == j);
        end
    end
    err_tr = 100 * sum(pred_tr ~= y_tr) / Ntr;
    err_te = 100 * sum(pred_te ~= y_te) / Nte;
end
