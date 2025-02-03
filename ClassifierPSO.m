function ClassificationSol = ClassifierPSO(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry
    if isSymmetry
        DamageLocation = GroupSymmetricalElements(DamageLocation, Symmetry);
    end

    % Convert labels to integers
    [uniqueClasses, ~, Y] = unique(DamageLocation);
    numClasses = length(uniqueClasses);

    % Split data
    [trainInd, testInd, valInd] = divideblock(length(Y), TrainRatio, TestRatio, 1-TrainRatio-TestRatio);
    X_train = NaturalFrequancy(trainInd, :);
    Y_train = Y(trainInd);
    X_test = NaturalFrequancy(testInd, :);
    Y_test = Y(testInd);
    X_val = NaturalFrequancy(valInd, :);
    Y_val = Y(valInd);

    % Normalize features and add bias
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_train_norm = [X_train_norm, ones(size(X_train_norm,1),1)]; % Bias term
    X_test_norm = (X_test - mu) ./ sigma;
    X_test_norm = [X_test_norm, ones(size(X_test_norm,1),1)];
    X_val_norm = (X_val - mu) ./ sigma;
    X_val_norm = [X_val_norm, ones(size(X_val_norm,1),1)];

    % Define fitness function for classification
    function error = FitnessFunction(weights)
        weightsMatrix = reshape(weights, numClasses, []);
        scores = X_train_norm * weightsMatrix';
        [~, pred] = max(scores, [], 2);
        error = mean(pred ~= Y_train);
    end

    % Configure PSO
    nVars = numClasses * size(X_train_norm, 2);
    options = optimoptions('particleswarm', ...
        'MaxIterations', MaxIterations, ...
        'SwarmSize', SwarmSize, ...
        'Display', 'iter');
    [bestWeights, ~] = particleswarm(@FitnessFunction, nVars, -10, 10, options);

    % Predict classes
    weightsMatrix = reshape(bestWeights, numClasses, []);
    scores_train = X_train_norm * weightsMatrix';
    [~, Y_train_pred] = max(scores_train, [], 2);
    scores_test = X_test_norm * weightsMatrix';
    [~, Y_test_pred] = max(scores_test, [], 2);
    scores_val = X_val_norm * weightsMatrix';
    [~, Y_val_pred] = max(scores_val, [], 2);

    % Convert predictions to original labels
    Y_train_pred = uniqueClasses(Y_train_pred);
    Y_test_pred = uniqueClasses(Y_test_pred);
    Y_val_pred = uniqueClasses(Y_val_pred);

    % Calculate accuracy
    trainAcc = mean(Y_train_pred == DamageLocation(trainInd));
    testAcc = mean(Y_test_pred == DamageLocation(testInd));
    valAcc = mean(Y_val_pred == DamageLocation(valInd));

    % Print results
    fprintf('\n--- PSO Classifier Results ---\n');
    fprintf('Training Accuracy:    %.2f%%\n', trainAcc * 100);
    fprintf('Testing Accuracy:     %.2f%%\n', testAcc * 100);
    fprintf('Validation Accuracy:  %.2f%%\n', valAcc * 100);
    fprintf('Confusion Matrix (Test Set):\n');
    disp(confusionmat(DamageLocation(testInd), Y_test_pred));

    % Prepare output
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

% Symmetry grouping function
function groupedLabels = GroupSymmetricalElements(labels, Symmetry)
    groupedLabels = labels;
    for i = 1:size(Symmetry,1)
        for j = 2:size(Symmetry,2)
            groupedLabels(labels == Symmetry(i,j)) = Symmetry(i,1);
        end
    end
end