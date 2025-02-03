function RegressionSol = RegressionPSOGA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
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

    % Define fitness function (MSE with output clamping)
    function mse = FitnessFunction(weights)
        Y_pred = X_train_norm * weights';
        Y_pred = max(min(Y_pred, max(Y_train)), min(Y_train)); % Clamp predictions
        mse = mean((Y_pred - Y_train).^2);
    end

    % Hybrid PSO-GA parameters
    nVars = size(X_train_norm, 2);
    lb = -3 * ones(1, nVars);  % Reduced bounds
    ub = 3 * ones(1, nVars);   % Reduced bounds
    velocityMax = 0.1 * (ub - lb); % Velocity clamping

    % Run hybrid PSO-GA
    [bestWeights, ~, fitnessHistory] = PSOGAHybrid(@FitnessFunction, nVars, lb, ub, MaxIterations, SwarmSize);

    % Predict using optimized weights
    Y_train_pred = X_train_norm * bestWeights';
    Y_test_pred = X_test_norm * bestWeights';
    Y_val_pred = X_val_norm * bestWeights';

    % Calculate metrics
    [trainRMSE, trainR2] = calculateMetrics(Y_train, Y_train_pred);
    [testRMSE, testR2] = calculateMetrics(Y_test, Y_test_pred);
    [valRMSE, valR2] = calculateMetrics(Y_val, Y_val_pred);

    % Print results
    fprintf('\n--- PSO-GA Regression Results ---\n');
    fprintf('Training RMSE:   %.4f\n', trainRMSE);
    fprintf('Training R²:     %.4f\n', trainR2);
    fprintf('Testing RMSE:    %.4f\n', testRMSE);
    fprintf('Testing R²:      %.4f\n', testR2);
    fprintf('Validation RMSE: %.4f\n', valRMSE);
    fprintf('Validation R²:   %.4f\n', valR2);

    % Plot convergence
    figure;
    plot(fitnessHistory, 'LineWidth', 2);
    title('Fitness Convergence');
    xlabel('Iteration');
    ylabel('MSE');
    grid on;

    % Prepare output structure
    RegressionSol = struct(...
        'TrainPredictions', Y_train_pred, ...
        'TestPredictions', Y_test_pred, ...
        'ValidationPredictions', Y_val_pred, ...
        'TrainRMSE', trainRMSE, 'TestRMSE', testRMSE, 'ValidationRMSE', valRMSE, ...
        'TrainR2', trainR2, 'TestR2', testR2, 'ValidationR2', valR2, ...
        'BestWeights', bestWeights, ...
        'FitnessHistory', fitnessHistory, ...
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