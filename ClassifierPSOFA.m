function ClassificationSol = ClassifierPSOFA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
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

    % Hybrid PSO-FA parameters
    nVars = numClasses * size(X_train_norm, 2); % Total weights = classes × (features + bias)
    lb = -10 * ones(1, nVars); % Lower bounds
    ub = 10 * ones(1, nVars);  % Upper bounds

    % Run hybrid PSO-FA
    [bestWeights, ~] = PSOFAHybrid(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

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
    fprintf('\n--- PSO-FA Classifier Results ---\n');
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

    % Nested helper functions
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

            % Update personal and global bests
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

    % Symmetry grouping helper function (nested)
    function groupedLabels = GroupSymmetricalElements(labels, Symmetry)
        groupedLabels = labels;
        for i = 1:size(Symmetry,1)
            for j = 2:size(Symmetry,2)
                groupedLabels(labels == Symmetry(i,j)) = Symmetry(i,1);
            end
        end
    end

end % End of main function