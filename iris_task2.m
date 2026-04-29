%% TTT4275 The Iris Task. Part 2: Features and Linear Separability
% Equations used (compendium "Part III Classification"):
%   Eq. 19 : MSE = (1/2) * sum_k (g_k - t_k)^T (g_k - t_k)
%   Eq. 20 : g_ik = 1 / (1 + exp(-z_ik)),  z_k = W*x_k
%   Eq. 22 : grad_W MSE = sum_k [(g_k - t_k) o g_k o (1 - g_k)] x_k^T
%   Eq. 23 : W(m) = W(m-1) - alpha * grad_W MSE
%
% The first 30 samples per class are used for training and the last 20
% for testing.

clear; close all; clc;

% Load the Iris dataset
x1all = load('class_1', '-ascii');   % Setosa     (50 x 4)
x2all = load('class_2', '-ascii');   % Versicolor (50 x 4)
x3all = load('class_3', '-ascii');   % Virginica  (50 x 4)

C  = 3;
D  = size(x1all, 2);
Ni = 50;

feature_names = {'Sepal Length','Sepal Width','Petal Length','Petal Width'};

alpha    = 0.005;
max_iter = 3000;

%Task 2a: histograms for each feature / class
all_data   = [x1all; x2all; x3all];
all_labels = [ones(Ni,1); 2*ones(Ni,1); 3*ones(Ni,1)];

figure('Name','Feature Histograms','Position',[100 100 1200 800]);
colors      = {'r','g','b'};
class_names = {'Setosa','Versicolor','Virginica'};
for f = 1:D
    subplot(2,2,f); hold on;
    for c = 1:C
        histogram(all_data(all_labels==c, f), 15, ...
                  'FaceColor', colors{c}, 'FaceAlpha', 0.4, ...
                  'EdgeColor','none');
    end
    hold off; 
    grid on;
    legend(class_names,'Location','best');
    title(feature_names{f});
    xlabel('Value [cm]'); ylabel('Count');
end

sgtitle('Feature histograms by class');
saveas(gcf,'feature_histograms.png');

%Histogram overlap score (smaller score = less overlap)
overlap = zeros(1,D);
for f = 1:D
    edges = linspace(min(all_data(:,f)), max(all_data(:,f)), 31);
    s = 0;
    for c1 = 1:C
        for c2 = c1+1:C
            h1 = histcounts(all_data(all_labels==c1,f), edges, 'Normalization','probability');
            h2 = histcounts(all_data(all_labels==c2,f), edges, 'Normalization','probability');
            s  = s + sum(min(h1,h2));
        end
    end
    overlap(f) = s;
end

fprintf('Histogram overlap per feature (higher = less discriminative):\n');
for f = 1:D
    fprintf('  %-13s : %.4f\n', feature_names{f}, overlap(f));
end

%Remove features in order of most-overlapping first
[~, rm_order] = sort(overlap, 'descend');
fprintf('\nRemoval order (most -> least overlapping):\n');
for i = 1:D
    fprintf('  %d. %s\n', i, feature_names{rm_order(i)});
end

%Train/test with decreasing number of features
experiments = cell(1,4);
experiments{1} = 1:D;                      % all features
experiments{2} = setdiff(1:D, rm_order(1),   'stable');
experiments{3} = setdiff(1:D, rm_order(1:2), 'stable');
experiments{4} = setdiff(1:D, rm_order(1:3), 'stable');

results = struct();
%Train linear classifier with augmented feature set
for e = 1:4
    feat = experiments{e};
    fprintf('\n=========================================================\n');
    fprintf('Experiment %d : %d features (%s)\n', e, numel(feat), ...
            strjoin(feature_names(feat), ', '));
    fprintf('=========================================================\n');

    [~, ~, cTr, eTr, cTe, eTe] = ...
        trainLinearMSE(x1all(:,feat), x2all(:,feat), x3all(:,feat), ...
                       1:30, 31:50, C, alpha, max_iter);

    fprintf('Training confusion matrix:\n'); disp(cTr);
    fprintf('Training error rate : %.2f %%\n\n', eTr);
    fprintf('Test confusion matrix:\n');     disp(cTe);
    fprintf('Test error rate     : %.2f %%\n', eTe);

    results(e).feats  = feat;
    results(e).names  = feature_names(feat);
    results(e).cTr    = cTr;
    results(e).eTr    = eTr;
    results(e).cTe    = cTe;
    results(e).eTe    = eTe;
end

%Summary table
fprintf('\n=========================================================\n');
fprintf('SUMMARY : error rates vs. number of features\n');
fprintf('=========================================================\n');
fprintf('# feat   Train err   Test err    Features\n');
for e = 1:4
    fprintf('  %d       %6.2f %%    %6.2f %%   (%s)\n', ...
        numel(results(e).feats), results(e).eTr, results(e).eTe, ...
        strjoin(results(e).names, ', '));
end


%Helper: same as in Part 1, Eq. 19-23 of the compendium.
function [W, mse_hist, conf_tr, err_tr, conf_te, err_te] = ...
    trainLinearMSE(x1all, x2all, x3all, trIdx, teIdx, C, alpha, max_iter)

    D = size(x1all, 2);

    X_tr = [x1all(trIdx,:); x2all(trIdx,:); x3all(trIdx,:)];
    X_te = [x1all(teIdx,:); x2all(teIdx,:); x3all(teIdx,:)];
    Ntr  = size(X_tr, 1);    Nte = size(X_te, 1);
    Ntri = length(trIdx);    Ntei = length(teIdx);

    y_tr = [ones(Ntri,1); 2*ones(Ntri,1); 3*ones(Ntri,1)];
    y_te = [ones(Ntei,1); 2*ones(Ntei,1); 3*ones(Ntei,1)];

    T_tr = zeros(Ntr, C);
    for k = 1:Ntr, T_tr(k, y_tr(k)) = 1; end

    Xa_tr = [X_tr, ones(Ntr, 1)];
    Xa_te = [X_te, ones(Nte, 1)];

    rng(42);
    W = 0.01 * randn(D+1, C);

    mse_hist = zeros(max_iter,1);
    for m = 1:max_iter
        Z = Xa_tr * W;                         % z_k = W * x_k
        G = 1 ./ (1 + exp(-Z));                % Eq. 20
        E = G - T_tr;
        mse_hist(m) = 0.5 * sum(E(:).^2);      % Eq. 19
        grad_W = Xa_tr' * ( E .* G .* (1 - G) );% Eq. 22
        W = W - alpha * grad_W;                % Eq. 23
    end

    [~, pred_tr] = max(Xa_tr * W, [], 2);
    [~, pred_te] = max(Xa_te * W, [], 2);

    conf_tr = zeros(C,C);  conf_te = zeros(C,C);
    for i = 1:C
        for j = 1:C
            conf_tr(i,j) = sum(pred_tr(y_tr == i) == j);
            conf_te(i,j) = sum(pred_te(y_te == i) == j);
        end
    end
    err_tr = 100 * sum(pred_tr ~= y_tr) / Ntr;
    err_te = 100 * sum(pred_te ~= y_te) / Nte;
end
