function ClassificationSol = ClassifierGA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry grouping
    if isSymmetry
        DamageLocation = GroupSymmetricalElements(DamageLocation, Symmetry);
    end

    % Convert labels to categorical integers
    [uniqueClasses, ~, classIndices] = unique(DamageLocation);
    numClasses = length(uniqueClasses);
    Y = classIndices;

    % Split data into training, testing, and validation sets
    [trainInd, testInd, valInd] = divideblock(length(Y), TrainRatio, TestRatio, 1-TrainRatio-TestRatio);

    X_train = NaturalFrequancy(trainInd, :);
    Y_train = Y(trainInd);
    X_test = NaturalFrequancy(testInd, :);
    Y_test = Y(testInd);
    X_val = NaturalFrequancy(valInd, :);
    Y_val = Y(valInd);

    % Normalize features
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ sigma;
    X_val_norm = (X_val - mu) ./ sigma;

    % Define fitness function for GA
    fitnessFcn = @(weights) FitnessFunction(weights, X_train_norm, Y_train, numClasses);

    % Configure GA options
    options = optimoptions('ga', ...
        'MaxGenerations', MaxIterations, ...
        'PopulationSize', SwarmSize, ...
        'Display', 'off');

    % Number of weights (features + 1 bias per class for one-vs-all)
    nFeatures = size(X_train_norm, 2);
    nVars = (nFeatures + 1) * numClasses;

    % Run GA
    [bestWeights, ~] = ga(fitnessFcn, nVars, [], [], [], [], [], [], [], options);

    % Predict using optimized weights
    Y_train_pred = PredictClasses(bestWeights, X_train_norm, numClasses);
    Y_test_pred = PredictClasses(bestWeights, X_test_norm, numClasses);
    Y_val_pred = PredictClasses(bestWeights, X_val_norm, numClasses);

    % Convert back to original class labels
    Y_train_pred = uniqueClasses(Y_train_pred);
    Y_test_pred = uniqueClasses(Y_test_pred);
    Y_val_pred = uniqueClasses(Y_val_pred);

    % Calculate accuracy
    trainAcc = mean(Y_train_pred == DamageLocation(trainInd));
    testAcc = mean(Y_test_pred == DamageLocation(testInd));
    valAcc = mean(Y_val_pred == DamageLocation(valInd));

    % Print results
    fprintf('\n--- GA Classifier Results ---\n');
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

function error = FitnessFunction(weights, X, Y, numClasses)
    % Reshape weights into a matrix (numClasses x (nFeatures + 1))
    nFeatures = size(X, 2);
    weightsMatrix = reshape(weights, numClasses, nFeatures + 1);

    % Predict classes using one-vs-all linear model
    scores = X * weightsMatrix(:, 1:end-1)' + weightsMatrix(:, end)';
    [~, pred] = max(scores, [], 2);

    % Calculate error (minimize misclassification rate)
    error = 1 - mean(pred == Y);
end

function pred = PredictClasses(weights, X, numClasses)
    nFeatures = size(X, 2);
    weightsMatrix = reshape(weights, numClasses, nFeatures + 1);
    scores = X * weightsMatrix(:, 1:end-1)' + weightsMatrix(:, end)';
    [~, pred] = max(scores, [], 2);
end