function RegressionSol = RegressionPSOGWO(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
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

    % Hybrid PSO-GWO parameters
    nVars = size(X_train_norm, 2); % Number of variables (features + bias)
    lb = -10 * ones(1, nVars);     % Lower bounds
    ub = 10 * ones(1, nVars);      % Upper bounds

    % Run hybrid PSO-GWO
    [bestWeights, ~] = PSOGWO(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

    % Predict using optimized weights
    Y_train_pred = X_train_norm * bestWeights';
    Y_test_pred = X_test_norm * bestWeights';
    Y_val_pred = X_val_norm * bestWeights';

    % Calculate metrics
    [trainRMSE, trainR2] = calculateMetrics(Y_train, Y_train_pred);
    [testRMSE, testR2] = calculateMetrics(Y_test, Y_test_pred);
    [valRMSE, valR2] = calculateMetrics(Y_val, Y_val_pred);

    % Print results
    fprintf('\n--- PSO-GWO Regression Results ---\n');
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

    % Hybrid PSO-GWO Optimizer
    function [alpha_pos, alpha_score] = PSOGWO(fitnessFcn, dim, lb, ub, maxIter, swarmSize)
        % Initialize population
        positions = rand(swarmSize, dim) .* (ub - lb) + lb;
        velocities = rand(swarmSize, dim) .* (ub - lb) * 0.1;
        
        % Initialize PSO parameters
        w = 0.729; % Inertia weight
        c1 = 1.49445; % Cognitive coefficient
        c2 = 1.49445; % Social coefficient
        
        % Initialize GWO parameters
        alpha_pos = zeros(1, dim);
        alpha_score = inf;
        beta_pos = zeros(1, dim);
        beta_score = inf;
        delta_pos = zeros(1, dim);
        delta_score = inf;

        for iter = 1:maxIter
            a = 2 - iter * (2 / maxIter); % Linearly decreases from 2 to 0
            
            % Evaluate fitness and update alpha, beta, delta wolves
            for i = 1:swarmSize
                fitness = fitnessFcn(positions(i, :));
                
                % Update alpha, beta, delta
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
            
            % Update positions using PSO and GWO
            for i = 1:swarmSize
                % PSO velocity update
                velocities(i, :) = w * velocities(i, :) + ...
                    c1 * rand() * (alpha_pos - positions(i, :)) + ...
                    c2 * rand() * (positions(i, :) - mean(positions));
                
                % GWO position update
                A1 = 2 * a * rand() - a;
                C1 = 2 * rand();
                D_alpha = abs(C1 * alpha_pos - positions(i, :));
                X1 = alpha_pos - A1 * D_alpha;
                
                A2 = 2 * a * rand() - a;
                C2 = 2 * rand();
                D_beta = abs(C2 * beta_pos - positions(i, :));
                X2 = beta_pos - A2 * D_beta;
                
                A3 = 2 * a * rand() - a;
                C3 = 2 * rand();
                D_delta = abs(C3 * delta_pos - positions(i, :));
                X3 = delta_pos - A3 * D_delta;
                
                % Combine PSO and GWO updates
                positions(i, :) = positions(i, :) + velocities(i, :) + (X1 + X2 + X3) / 3;
                positions(i, :) = min(max(positions(i, :), lb), ub);
            end
        end
    end
end