function ClassificationSol = ClassifierFA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry grouping if enabled
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
    
    % Add bias term
    X_train_norm = [X_train_norm ones(size(X_train_norm,1),1)];
    X_test_norm = [X_test_norm ones(size(X_test_norm,1),1)];
    X_val_norm = [X_val_norm ones(size(X_val_norm,1),1)];
    
    % FA parameters
    nVar = size(X_train_norm, 2) * numClasses; % Weight matrix elements
    alpha = 0.5;        % Increased randomness factor
    beta0 = 2.0;        % Higher base attractiveness
    gamma = 0.01;       % Reduced absorption coefficient
    lb = -20;           % Wider search bounds
    ub = 20;
    MaxIterations = 100;% Increased iterations
    SwarmSize = 50;     % Larger population
    
    % Initialize fireflies (each represents a weight matrix)
    fireflies = lb + (ub - lb) * rand(SwarmSize, nVar);
    fitness = zeros(SwarmSize, 1);
    
    % FA optimization loop
    for iter = 1:MaxIterations
        % Evaluate all fireflies
        for i = 1:SwarmSize
            % Reshape weights to matrix form
            W = reshape(fireflies(i,:), [size(X_train_norm,2), numClasses]);
            
            % Calculate scores and probabilities
            scores = X_train_norm * W;
            exp_scores = exp(scores - max(scores, [], 2));
            probs = exp_scores ./ sum(exp_scores, 2);
            
            % Calculate cross-entropy loss
            correct_probs = probs(sub2ind(size(probs), 1:size(probs,1), Y_train'));
            fitness(i) = -mean(log(correct_probs + eps));
        end
        
        % Update firefly positions
        new_fireflies = fireflies;
        for i = 1:SwarmSize
            for j = 1:SwarmSize
                if fitness(j) < fitness(i)
                    % Calculate distance
                    r = norm(fireflies(i,:) - fireflies(j,:));
                    
                    % Update movement
                    beta = beta0 * exp(-gamma * r^2);
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
        fprintf('Iteration %d: Best Loss = %.4f\n', iter, min(fitness));
    end
    
    % Find best solution
    [~, bestIdx] = min(fitness);
    bestW = reshape(fireflies(bestIdx,:), [size(X_train_norm,2), numClasses]);
    
    % Calculate predictions for all datasets
    [~, Y_train_pred] = max(X_train_norm * bestW, [], 2);
    [~, Y_test_pred] = max(X_test_norm * bestW, [], 2);
    [~, Y_val_pred] = max(X_val_norm * bestW, [], 2);
    
    % Convert back to original class labels
    Y_train_pred = uniqueClasses(Y_train_pred);
    Y_test_pred = uniqueClasses(Y_test_pred);
    Y_val_pred = uniqueClasses(Y_val_pred);
    
    % Calculate accuracy
    trainAcc = mean(Y_train_pred == DamageLocation(trainInd));
    testAcc = mean(Y_test_pred == DamageLocation(testInd));
    valAcc = mean(Y_val_pred == DamageLocation(valInd));
    
    % Prepare output structure
    ClassificationSol = struct(...
        'TrainOutput', Y_train_pred, ...
        'TestOutput', Y_test_pred, ...
        'ValidationOutput', Y_val_pred, ...
        'TrainAccuracy', trainAcc, ...
        'TestAccuracy', testAcc, ...
        'ValidationAccuracy', valAcc, ...
        'ConfusionMatrix', confusionmat(DamageLocation(testInd), Y_test_pred), ...
        'BestWeights', bestW, ...
        'ClassLabels', uniqueClasses, ...
        'NormalizationParams', struct('mu', mu, 'sigma', sigma) ...
    );
end

function groupedLabels = GroupSymmetricalElements(labels, Symmetry)
    groupedLabels = labels;
    for i = 1:size(Symmetry,1)
        for j = 2:size(Symmetry,2)
            groupedLabels(labels == Symmetry(i,j)) = Symmetry(i,1);
        end
    end
end