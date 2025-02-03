function ClassificationSol = ClassifierGWO(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry grouping if enabled
    if isSymmetry
        DamageLocation = GroupSymmetricalElements(DamageLocation, Symmetry);
    end

    % Convert labels to categorical indices (1, 2, ..., C)
    [uniqueClasses, ~, Y] = unique(DamageLocation);
    numClasses = length(uniqueClasses);

    % Split data into training, testing, and validation sets
    [trainInd, testInd, valInd] = divideblock(length(Y), TrainRatio, TestRatio, 1-TrainRatio-TestRatio);
    X_train = NaturalFrequancy(trainInd, :);
    Y_train = Y(trainInd);
    X_test = NaturalFrequancy(testInd, :);
    Y_test = Y(testInd);
    X_val = NaturalFrequancy(valInd, :);
    Y_val = Y(valInd);

    % Normalize features and add bias term
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_train_norm = [X_train_norm, ones(size(X_train_norm,1),1)]; % Bias term
    X_test_norm = (X_test - mu) ./ sigma;
    X_test_norm = [X_test_norm, ones(size(X_test_norm,1),1)];
    X_val_norm = (X_val - mu) ./ sigma;
    X_val_norm = [X_val_norm, ones(size(X_val_norm,1),1)];

    % Define fitness function (classification error)
    function error = FitnessFunction(weights)
        weightsMatrix = reshape(weights, numClasses, []);
        scores = X_train_norm * weightsMatrix';
        [~, pred] = max(scores, [], 2);
        error = mean(pred ~= Y_train);
    end

    % Configure GWO parameters
    nVars = numClasses * size(X_train_norm, 2); % Total weights = classes × (features + bias)
    lb = -10 * ones(1, nVars); % Lower bounds
    ub = 10 * ones(1, nVars);  % Upper bounds

    % Run GWO
    [bestWeights, ~] = GWO(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

    % Predict classes using optimized weights
    weightsMatrix = reshape(bestWeights, numClasses, []);
    scores_train = X_train_norm * weightsMatrix';
    [~, Y_train_pred] = max(scores_train, [], 2);
    scores_test = X_test_norm * weightsMatrix';
    [~, Y_test_pred] = max(scores_test, [], 2);
    scores_val = X_val_norm * weightsMatrix';
    [~, Y_val_pred] = max(scores_val, [], 2);

    % Convert indices back to original labels
    Y_train_pred = uniqueClasses(Y_train_pred);
    Y_test_pred = uniqueClasses(Y_test_pred);
    Y_val_pred = uniqueClasses(Y_val_pred);

    % Calculate accuracy
    trainAcc = mean(Y_train_pred == DamageLocation(trainInd));
    testAcc = mean(Y_test_pred == DamageLocation(testInd));
    valAcc = mean(Y_val_pred == DamageLocation(valInd));

    % Print results
    fprintf('\n--- GWO Classifier Results ---\n');
    fprintf('Training Accuracy:    %.2f%%\n', trainAcc * 100);
    fprintf('Testing Accuracy:     %.2f%%\n', testAcc * 100);
    fprintf('Validation Accuracy:  %.2f%%\n', valAcc * 100);
    fprintf('Confusion Matrix (Test Set):\n');
    disp(confusionmat(DamageLocation(testInd), Y_test_pred));

    % Prepare output structure
    ClassificationSol = struct(...
        'TrainOutput', Y_train_pred, ...
        'TestOutput', Y_test_pred, ...
        'ValidationOutput', Y_val_pred, ...
        'TrainAccuracy', trainAcc, ...
        'TestAccuracy', testAcc, ...
        'ValidationAccuracy', valAcc, ...
        'ConfusionMatrix', confusionmat(DamageLocation(testInd), Y_test_pred), ...
        'BestWeights', bestWeights, ...
        'ClassLabels', uniqueClasses, ...
        'NormalizationParams', struct('mu', mu, 'sigma', sigma), ...
        'TestIndices', testInd ...
    );
end

% Grey Wolf Optimizer (GWO) Implementation (Nested)
function [alpha_pos, alpha_score] = GWO(fitnessFcn, dim, lb, ub, maxIter, swarmSize)
    alpha_pos = zeros(1, dim);
    alpha_score = inf;
    beta_pos = zeros(1, dim);
    beta_score = inf;
    delta_pos = zeros(1, dim);
    delta_score = inf;

    % Initialize population within bounds
    positions = rand(swarmSize, dim) .* (ub - lb) + lb;

    for iter = 1:maxIter
        a = 2 - iter * (2 / maxIter); % Linearly decreases from 2 to 0

        for i = 1:swarmSize
            % Evaluate fitness for the current wolf
            fitness = fitnessFcn(positions(i, :));

            % Update alpha, beta, and delta wolves
            if fitness < alpha_score
                alpha_score = fitness;
                alpha_pos = positions(i, :);
            elseif fitness < beta_score
                beta_score = fitness;
                beta_pos = positions(i, :);
            elseif fitness < delta_score
                delta_score = fitness;
                delta_pos = positions(i, :);
            end
        end

        % Update positions of all wolves
        for i = 1:swarmSize
            for j = 1:dim
                % Compute coefficients for alpha, beta, and delta
                A1 = 2 * a * rand() - a;
                C1 = 2 * rand();
                D_alpha = abs(C1 * alpha_pos(j) - positions(i, j));
                X1 = alpha_pos(j) - A1 * D_alpha;

                A2 = 2 * a * rand() - a;
                C2 = 2 * rand();
                D_beta = abs(C2 * beta_pos(j) - positions(i, j));
                X2 = beta_pos(j) - A2 * D_beta;

                A3 = 2 * a * rand() - a;
                C3 = 2 * rand();
                D_delta = abs(C3 * delta_pos(j) - positions(i, j));
                X3 = delta_pos(j) - A3 * D_delta;

                % Update the position of the current wolf
                positions(i, j) = (X1 + X2 + X3) / 3;
                positions(i, j) = min(max(positions(i, j), lb(j)), ub(j));
            end
        end
    end
end

% Symmetry grouping helper function
function groupedLabels = GroupSymmetricalElements(labels, Symmetry)
    groupedLabels = labels;
    for i = 1:size(Symmetry,1)
        for j = 2:size(Symmetry,2)
            groupedLabels(labels == Symmetry(i,j)) = Symmetry(i,1);
        end
    end
end