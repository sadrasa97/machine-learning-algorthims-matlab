function RegressionSol = RegressionPSOFA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
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
    X_train_norm = [X_train_norm, ones(size(X_train_norm,1),1)]; % Bias term
    X_test_norm = (X_test - mu) ./ sigma;
    X_test_norm = [X_test_norm, ones(size(X_test_norm,1),1)];
    X_val_norm = (X_val - mu) ./ sigma;
    X_val_norm = [X_val_norm, ones(size(X_val_norm,1),1)];

    % Define fitness function (MSE)
    function mse = FitnessFunction(weights)
        Y_pred = X_train_norm * weights';
        mse = mean((Y_pred - Y_train).^2);
    end

    % Hybrid PSO-FA parameters
    nVars = size(X_train_norm, 2); % Number of variables (features + bias)
    lb = -10 * ones(1, nVars);     % Lower bounds
    ub = 10 * ones(1, nVars);      % Upper bounds

    % Run hybrid PSO-FA
    [bestWeights, ~] = PSOFAHybrid(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

    % Predict using optimized weights
    Y_train_pred = X_train_norm * bestWeights';
    Y_test_pred = X_test_norm * bestWeights';
    Y_val_pred = X_val_norm * bestWeights';

    % Calculate metrics
    [trainRMSE, trainR2] = calculateMetrics(Y_train, Y_train_pred);
    [testRMSE, testR2] = calculateMetrics(Y_test, Y_test_pred);
    [valRMSE, valR2] = calculateMetrics(Y_val, Y_val_pred);

    % Print results
    fprintf('\n--- PSO-FA Regression Results ---\n');
    fprintf('Training RMSE:   %.4f\n', trainRMSE);
    fprintf('Training R�:     %.4f\n', trainR2);
    fprintf('Testing RMSE:    %.4f\n', testRMSE);
    fprintf('Testing R�:      %.4f\n', testR2);
    fprintf('Validation RMSE: %.4f\n', valRMSE);
    fprintf('Validation R�:   %.4f\n', valR2);

    % Prepare output structure
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

    % Nested helper functions
    function [rmse, r2] = calculateMetrics(Y_true, Y_pred)
        rmse = sqrt(mean((Y_true - Y_pred).^2));
        ss_total = sum((Y_true - mean(Y_true)).^2);
        ss_residual = sum((Y_true - Y_pred).^2);
        r2 = 1 - (ss_residual / ss_total);
    end

    function [bestSolution, bestFitness] = PSOFAHybrid(fitnessFcn, nVars, lb, ub, maxIter, swarmSize)
        % Initialize swarm
        positions = rand(swarmSize, nVars) .* (ub - lb) + lb;
        velocities = zeros(swarmSize, nVars);
        personalBest = positions;
        personalBestFitness = inf(swarmSize, 1);
        globalBestFitness = inf;
        globalBestPosition = zeros(1, nVars);

        % PSO parameters
        w = 0.729; c1 = 1.49445; c2 = 1.49445;

        % FA parameters
        beta0 = 1; gamma = 0.1; alpha = 0.2;

        for iter = 1:maxIter
            % PSO phase (first half of swarm)
            for i = 1:floor(swarmSize/2)
                % Update velocity and position
                velocities(i,:) = w * velocities(i,:) + ...
                    c1 * rand() * (personalBest(i,:) - positions(i,:)) + ...
                    c2 * rand() * (globalBestPosition - positions(i,:));
                positions(i,:) = positions(i,:) + velocities(i,:);
                positions(i,:) = min(max(positions(i,:), lb), ub);
            end

            % FA phase (second half of swarm)
            for i = floor(swarmSize/2)+1:swarmSize
                for j = 1:swarmSize
                    if fitnessFcn(positions(j,:)) < fitnessFcn(positions(i,:))
                        r = norm(positions(i,:) - positions(j,:));
                        beta = beta0 * exp(-gamma * r^2);
                        positions(i,:) = positions(i,:) + ...
                            beta * (positions(j,:) - positions(i,:)) + ...
                            alpha * (rand(1,nVars) - 0.5);
                        positions(i,:) = min(max(positions(i,:), lb), ub);
                    end
                end
            end

            % Evaluate fitness and update bests
            for i = 1:swarmSize
                currentFitness = fitnessFcn(positions(i,:));
                if currentFitness < personalBestFitness(i)
                    personalBestFitness(i) = currentFitness;
                    personalBest(i,:) = positions(i,:);
                end
                if currentFitness < globalBestFitness
                    globalBestFitness = currentFitness;
                    globalBestPosition = positions(i,:);
                end
            end
        end
        bestSolution = globalBestPosition;
        bestFitness = globalBestFitness;
    end
end
