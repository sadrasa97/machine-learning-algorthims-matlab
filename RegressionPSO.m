function RegressionSol = RegressionPSO(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
    % Split data into training, testing, and validation sets
    [trainInd, testInd, valInd] = divideblock(size(NaturalFrequancy,1), TrainRatio, TestRatio, 1-TrainRatio-TestRatio);
    
    X_train = NaturalFrequancy(trainInd, :);
    Y_train = DamageRatio(trainInd);
    X_test = NaturalFrequancy(testInd, :);
    Y_test = DamageRatio(testInd);
    X_val = NaturalFrequancy(valInd, :);
    Y_val = DamageRatio(valInd);

    % Normalize features and add bias term
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_train_norm = [X_train_norm, ones(size(X_train_norm,1),1)]; % Add bias
    X_test_norm = (X_test - mu) ./ sigma;
    X_test_norm = [X_test_norm, ones(size(X_test_norm,1),1)];
    X_val_norm = (X_val - mu) ./ sigma;
    X_val_norm = [X_val_norm, ones(size(X_val_norm,1),1)];

    % Define fitness function (nested)
    function mse = FitnessFunction(weights)
        Y_pred = X_train_norm * weights';
        mse = mean((Y_pred - Y_train).^2);
    end

    % Configure PSO with bounds
    nVars = size(X_train_norm, 2);
    options = optimoptions('particleswarm', ...
        'MaxIterations', MaxIterations, ...
        'SwarmSize', SwarmSize, ...
        'Display', 'iter');
    [bestWeights, ~] = particleswarm(@FitnessFunction, nVars, -10, 10, options);

    % Predict using optimized weights
    Y_train_pred = X_train_norm * bestWeights';
    Y_test_pred = X_test_norm * bestWeights';
    Y_val_pred = X_val_norm * bestWeights';

    % Calculate metrics
    [trainRMSE, trainR2] = calculateMetrics(Y_train, Y_train_pred);
    [testRMSE, testR2] = calculateMetrics(Y_test, Y_test_pred);
    [valRMSE, valR2] = calculateMetrics(Y_val, Y_val_pred);

    % Print results
    fprintf('\n--- PSO Regression Results ---\n');
    fprintf('Training RMSE:   %.4f\n', trainRMSE);
    fprintf('Training R²:     %.4f\n', trainR2);
    fprintf('Testing RMSE:    %.4f\n', testRMSE);
    fprintf('Testing R²:      %.4f\n', testR2);
    fprintf('Validation RMSE: %.4f\n', valRMSE);
    fprintf('Validation R²:   %.4f\n', valR2);

    % Prepare output
    RegressionSol = struct(...
        'TrainPredictions', Y_train_pred, ...
        'TestPredictions', Y_test_pred, ...
        'ValidationPredictions', Y_val_pred, ...
        'TrainRMSE', trainRMSE, 'TestRMSE', testRMSE, 'ValidationRMSE', valRMSE, ...
        'TrainR2', trainR2, 'TestR2', testR2, 'ValidationR2', valR2, ...
        'BestWeights', bestWeights, ...
        'NormalizationParams', struct('mu', mu, 'sigma', sigma), ...
        'TestIndices', testInd ...
    );

    % Nested metric calculation
    function [rmse, r2] = calculateMetrics(Y_true, Y_pred)
        rmse = sqrt(mean((Y_true - Y_pred).^2));
        ss_total = sum((Y_true - mean(Y_true)).^2);
        ss_residual = sum((Y_true - Y_pred).^2);
        r2 = 1 - (ss_residual / ss_total);
    end
end