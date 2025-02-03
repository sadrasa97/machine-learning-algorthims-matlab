function ClassificationSol = ClassifierPSOGA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry grouping
    if isSymmetry
        DamageLocation = GroupSymmetricalElements(DamageLocation, Symmetry);
    end

    % Convert labels to categorical indices
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
    X_train_norm = [X_train_norm, ones(size(X_train_norm,1),1)];
    X_test_norm = (X_test - mu) ./ sigma;
    X_test_norm = [X_test_norm, ones(size(X_test_norm,1),1)];
    X_val_norm = (X_val - mu) ./ sigma;
    X_val_norm = [X_val_norm, ones(size(X_val_norm,1),1)];

    % Class balancing weights
    classCounts = histcounts(Y_train, 1:numClasses+1); % Ensure bins match class labels
    classWeights = 1./(classCounts + eps);
    classWeights = classWeights / sum(classWeights); % Normalize weights

    % Fitness function with class balancing
    function error = FitnessFunction(weights)
        weightsMatrix = reshape(weights, numClasses, []);
        scores = X_train_norm * weightsMatrix';
        [~, pred] = max(scores, [], 2);
        
        % Identify misclassified samples
        misclassified = (pred ~= Y_train);
        
        % Get true classes of misclassified samples
        trueClasses = Y_train(misclassified);
        
        % Calculate weighted error
        if isempty(trueClasses)
            error = 0; % No misclassifications
        else
            error = sum(classWeights(trueClasses));
        end
    end

    % Hybrid parameters
    nVars = numClasses * size(X_train_norm, 2);
    lb = -3 * ones(1, nVars);
    ub = 3 * ones(1, nVars);
    velocityMax = 0.1 * (ub - lb);

    % Run optimization
    [bestWeights, ~, fitnessHistory] = PSOGAHybrid(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

    % Predict classes
    weightsMatrix = reshape(bestWeights, numClasses, []);
    scores_train = X_train_norm * weightsMatrix';
    [~, Y_train_pred] = max(scores_train, [], 2);
    scores_test = X_test_norm * weightsMatrix';
    [~, Y_test_pred] = max(scores_test, [], 2);
    scores_val = X_val_norm * weightsMatrix';
    [~, Y_val_pred] = max(scores_val, [], 2);

    % Convert indices to labels
    Y_train_pred = uniqueClasses(Y_train_pred);
    Y_test_pred = uniqueClasses(Y_test_pred);
    Y_val_pred = uniqueClasses(Y_val_pred);

    % Calculate accuracy
    trainAcc = mean(Y_train_pred == DamageLocation(trainInd));
    testAcc = mean(Y_test_pred == DamageLocation(testInd));
    valAcc = mean(Y_val_pred == DamageLocation(valInd));

    % Print results
    fprintf('\n--- PSO-GA Classifier Results ---\n');
    fprintf('Training Accuracy:    %.2f%%\n', trainAcc * 100);
    fprintf('Testing Accuracy:     %.2f%%\n', testAcc * 100);
    fprintf('Validation Accuracy:  %.2f%%\n', valAcc * 100);
    fprintf('Confusion Matrix (Test Set):\n');
    disp(confusionmat(DamageLocation(testInd), Y_test_pred));

    % Plot convergence
    figure;
    plot(fitnessHistory, 'LineWidth', 2);
    title('Fitness Convergence');
    xlabel('Iteration');
    ylabel('Weighted Classification Error');
    grid on;

    % Prepare output
    ClassificationSol = struct(...
        'TrainOutput', Y_train_pred, ...
        'TestOutput', Y_test_pred, ...
        'ValidationOutput', Y_val_pred, ...
        'TrainAccuracy', trainAcc, 'TestAccuracy', testAcc, 'ValidationAccuracy', valAcc, ...
        'ConfusionMatrix', confusionmat(DamageLocation(testInd), Y_test_pred), ...
        'BestWeights', bestWeights, ...
        'FitnessHistory', fitnessHistory, ...
        'ClassLabels', uniqueClasses, ...
        'NormalizationParams', struct('mu', mu, 'sigma', sigma), ...
        'TestIndices', testInd ...
    );

    function [bestSolution, bestFitness, fitnessHistory] = PSOGAHybrid(fitnessFcn, nVars, lb, ub, maxIter, swarmSize)
        % Initialize swarm
        positions = rand(swarmSize, nVars) .* (ub - lb) + lb;
        velocities = zeros(swarmSize, nVars);
        personalBest = positions;
        personalBestFitness = inf(swarmSize, 1);
        globalBestFitness = inf;
        globalBestPosition = zeros(1, nVars);
        fitnessHistory = zeros(maxIter, 1);

        % PSO parameters
        w = 0.729; c1 = 1.49445; c2 = 1.49445;

        for iter = 1:maxIter
            % PSO velocity update with clamping
            for i = 1:swarmSize
                % Update velocity
                velocities(i,:) = w * velocities(i,:) + ...
                    c1 * rand() * (personalBest(i,:) - positions(i,:)) + ...
                    c2 * rand() * (globalBestPosition - positions(i,:));
                
                % Apply velocity clamping
                velocities(i,:) = min(max(velocities(i,:), -velocityMax), velocityMax);
                
                % Update position
                positions(i,:) = positions(i,:) + velocities(i,:);
                positions(i,:) = min(max(positions(i,:), lb), ub);
            end

            % GA operations every 10 iterations
            if mod(iter, 10) == 0
                % Tournament selection
                parents = TournamentSelection(personalBest, personalBestFitness, 2);
                
                % SBX Crossover
                offspring = SBXCrossover(parents, lb, ub);
                
                % Polynomial Mutation
                offspring = PolynomialMutation(offspring, lb, ub);
                
                % Replace worst particles
                [~, sortedIndices] = sort(personalBestFitness, 'descend');
                idx = sortedIndices(1:size(offspring,1));
                personalBest(idx,:) = offspring;
                positions(idx,:) = offspring;
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
            
            fitnessHistory(iter) = globalBestFitness;
        end
        bestSolution = globalBestPosition;
        bestFitness = globalBestFitness;
    end

    function selected = TournamentSelection(population, fitness, k)
        n = size(population, 1);
        selected = zeros(k, size(population, 2));
        for i = 1:k
            candidates = randperm(n, 2);
            [~, idx] = min(fitness(candidates));
            selected(i,:) = population(candidates(idx),:);
        end
    end

    function offspring = SBXCrossover(parents, lb, ub)
        eta = 20;
        offspring = zeros(size(parents));
        for i = 1:size(parents,1)/2
            p1 = parents(2*i-1,:);
            p2 = parents(2*i,:);
            for j = 1:length(p1)
                if rand() < 0.7 % 70% crossover probability
                    u = rand();
                    if u <= 0.5
                        beta = (2*u)^(1/(eta+1));
                    else
                        beta = (1/(2*(1-u)))^(1/(eta+1));
                    end
                    offspring(2*i-1,j) = 0.5*((1+beta)*p1(j) + (1-beta)*p2(j));
                    offspring(2*i,j) = 0.5*((1-beta)*p1(j) + (1+beta)*p2(j));
                else
                    offspring(2*i-1,j) = p1(j);
                    offspring(2*i,j) = p2(j);
                end
            end
        end
        offspring = min(max(offspring, lb), ub);
    end

    function mutated = PolynomialMutation(offspring, lb, ub)
        eta = 20;
        mutationProb = 1/size(offspring,2);
        mutated = offspring;
        for i = 1:size(offspring,1)
            for j = 1:size(offspring,2)
                if rand() < mutationProb
                    delta = min(ub(j)-mutated(i,j), mutated(i,j)-lb(j))/(ub(j)-lb(j));
                    r = rand();
                    if r < 0.5
                        deltaq = (2*r + (1-2*r)*(1-delta)^(eta+1))^(1/(eta+1)) - 1;
                    else
                        deltaq = 1 - (2*(1-r) + 2*(r-0.5)*(1-delta)^(eta+1))^(1/(eta+1));
                    end
                    mutated(i,j) = mutated(i,j) + deltaq*(ub(j)-lb(j));
                end
            end
        end
        mutated = min(max(mutated, lb), ub);
    end
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