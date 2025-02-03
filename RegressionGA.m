function RegressionSol = RegressionGA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
    % Split data into training, testing, and validation sets
    [trainInd, testInd, valInd] = divideblock(size(NaturalFrequancy,1), TrainRatio, TestRatio, 1-TrainRatio-TestRatio);
    
    X_train = NaturalFrequancy(trainInd, :);
    Y_train = DamageRatio(trainInd);
    X_test = NaturalFrequancy(testInd, :);
    Y_test = DamageRatio(testInd);
    X_val = NaturalFrequancy(valInd, :);
    Y_val = DamageRatio(valInd);

    % Normalize features
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ sigma;
    X_val_norm = (X_val - mu) ./ sigma;

    % Define fitness function (minimize MSE)
    fitnessFcn = @(weights) mean((X_train_norm * weights(1:end-1)' + weights(end) - Y_train).^2);

    % Configure GA options
    options = optimoptions('ga', ...
        'MaxGenerations', MaxIterations, ...
        'PopulationSize', SwarmSize, ...
        'Display', 'off');

    % Number of variables: features + 1 bias term
    nVars = size(X_train_norm, 2) + 1;

    % Run GA
    [bestWeights, ~] = ga(fitnessFcn, nVars, [], [], [], [], [], [], [], options);

    % Predict using optimized weights
    Y_train_pred = X_train_norm * bestWeights(1:end-1)' + bestWeights(end);
    Y_test_pred = X_test_norm * bestWeights(1:end-1)' + bestWeights(end);
    Y_val_pred = X_val_norm * bestWeights(1:end-1)' + bestWeights(end);

    % Calculate performance metrics
    [trainRMSE, trainR2] = calculateMetrics(Y_train, Y_train_pred);
    [testRMSE, testR2] = calculateMetrics(Y_test, Y_test_pred);
    [valRMSE, valR2] = calculateMetrics(Y_val, Y_val_pred);

    % Print results
    fprintf('\n--- GA Regression Results ---\n');
    fprintf('Training RMSE:   %.4f\n', trainRMSE);
    fprintf('Training R²:     %.4f\n', trainR2);
    fprintf('Testing RMSE:    %.4f\n', testRMSE);
    fprintf('Testing R²:      %.4f\n', testR2);
    fprintf('Validation RMSE: %.4f\n', valRMSE);
    fprintf('Validation R²:   %.4f\n', valR2);

    % Prepare output structure
    RegressionSol = struct(...
        'TrainPredictions', Y_train_pred, ...
        'TestPredictions', Y_test_pred, ...
        'ValidationPredictions', Y_val_pred, ...
        'TrainRMSE', trainRMSE, ...
        'TestRMSE', testRMSE, ...
        'ValidationRMSE', valRMSE, ...
        'TrainR2', trainR2, ...
        'TestR2', testR2, ...
        'ValidationR2', valR2, ...
        'BestWeights', bestWeights, ...
        'NormalizationParams', struct('mu', mu, 'sigma', sigma), ...
        'TestIndices', testInd ...
    );

    % Nested helper function
    function [rmse, r2] = calculateMetrics(Y_true, Y_pred)
        rmse = sqrt(mean((Y_true - Y_pred).^2));
        ss_total = sum((Y_true - mean(Y_true)).^2);
        ss_residual = sum((Y_true - Y_pred).^2);
        r2 = 1 - (ss_residual / ss_total);
    end
end