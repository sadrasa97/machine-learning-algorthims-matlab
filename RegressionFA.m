function RegressionSol = RegressionFA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
    % Split data into training, testing, and validation sets
    [trainInd, testInd, valInd] = divideblock(size(NaturalFrequancy,1), TrainRatio, TestRatio, 1-TrainRatio-TestRatio);
    
    X_train = NaturalFrequancy(trainInd, :);
    Y_train = DamageRatio(trainInd);
    X_test = NaturalFrequancy(testInd, :);
    Y_test = DamageRatio(testInd);
    X_val = NaturalFrequancy(valInd, :);
    Y_val = DamageRatio(valInd);
    
    % Normalize features (critical for FA performance)
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ sigma;
    X_val_norm = (X_val - mu) ./ sigma;
    
    % Add bias term to features
    X_train_norm = [X_train_norm ones(size(X_train_norm,1),1)];
    X_test_norm = [X_test_norm ones(size(X_test_norm,1),1)];
    X_val_norm = [X_val_norm ones(size(X_val_norm,1),1)];
    
    % FA parameters
    nVar = size(X_train_norm, 2);  % Number of variables (coefficients)
    alpha = 0.25;                  % Randomness factor (0-1)
    beta0 = 1.0;                   % Base attractiveness
    gamma = 0.1;                   % Light absorption coefficient
    lb = -10;                      % Lower bound for coefficients
    ub = 10;                       % Upper bound for coefficients
    
    % Initialize fireflies
    fireflies = lb + (ub - lb) * rand(SwarmSize, nVar);
    fitness = zeros(SwarmSize, 1);
    
    % FA optimization loop
    for iter = 1:MaxIterations
        % Evaluate all fireflies
        for i = 1:SwarmSize
            Y_pred = X_train_norm * fireflies(i,:)';
            fitness(i) = mean((Y_pred - Y_train).^2);  % MSE
        end
        
        % Update firefly positions
        new_fireflies = fireflies;
        for i = 1:SwarmSize
            for j = 1:SwarmSize
                if fitness(j) < fitness(i)
                    % Calculate distance between fireflies
                    r = sqrt(sum((fireflies(i,:) - fireflies(j,:)).^2));
                    
                    % Update movement
                    beta = beta0 * exp(-gamma * r.^2);
                    randomness = alpha * (rand(1,nVar) - 0.5);
                    new_fireflies(i,:) = fireflies(i,:) + ...
                        beta * (fireflies(j,:) - fireflies(i,:)) + ...
                        randomness;
                    
                    % Apply bounds
                    new_fireflies(i,:) = max(new_fireflies(i,:), lb);
                    new_fireflies(i,:) = min(new_fireflies(i,:), ub);
                end
            end
        end
        fireflies = new_fireflies;
        
        % Display progress
        fprintf('Iteration %d: Best MSE = %.4f\n', iter, min(fitness));
    end
    
    % Find best solution
    [~, bestIdx] = min(fitness);
    bestWeights = fireflies(bestIdx, :);
    
    % Calculate predictions
    Y_train_pred = X_train_norm * bestWeights';
    Y_test_pred = X_test_norm * bestWeights';
    Y_val_pred = X_val_norm * bestWeights';
    
    % Calculate errors
    trainError = sqrt(mean((Y_train_pred - Y_train).^2));  % RMSE
    testError = sqrt(mean((Y_test_pred - Y_test).^2));
    valError = sqrt(mean((Y_val_pred - Y_val).^2));
    
    % Prepare output structure
    RegressionSol = struct(...
        'TrainOutput', Y_train_pred, ...
        'TestOutput', Y_test_pred, ...
        'ValidationOutput', Y_val_pred, ...
        'TrainError', trainError, ...
        'TestError', testError, ...
        'ValidationError', valError, ...
        'BestWeights', bestWeights, ...
        'NormalizationParams', struct('mu', mu, 'sigma', sigma) ...
    );
end