function ClassificationSol = ClassifierPSOGWO(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry grouping if enabled
    if isSymmetry
        DamageLocation = GroupSymmetricalElements(DamageLocation, Symmetry);
    end

    % Convert labels to categorical indices
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

    % Hybrid PSO-GWO parameters
    nVars = numClasses * size(X_train_norm, 2); % Total weights = classes × (features + bias)
    lb = -10 * ones(1, nVars); % Lower bounds
    ub = 10 * ones(1, nVars);  % Upper bounds

    % Run hybrid PSO-GWO
    [bestWeights, ~] = PSOGWO(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

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
    fprintf('\n--- PSO-GWO Classifier Results ---\n');
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

% Symmetry grouping helper function
function groupedLabels = GroupSymmetricalElements(labels, Symmetry)
    groupedLabels = labels;
    for i = 1:size(Symmetry,1)
        for j = 2:size(Symmetry,2)
            groupedLabels(labels == Symmetry(i,j)) = Symmetry(i,1);
        end
    end
end
