function bestSol = RegressionSVMPSOGAFA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)

    dim = 2;
    lb = [0.1, 0.001];  % lower bounds for [C, epsilon]
    ub = [1000, 1];     % upper bounds for [C, epsilon]
    
    %%% Prepare the data %%%
    X = NaturalFrequancy;
    Y = DamageRatio;
    N = size(X,1);
    idx = randperm(N);
    X = X(idx,:);
    Y = Y(idx);
    
    nTrain = floor(TrainRatio * N);
    nTest  = N - nTrain;
    Xtrain = X(1:nTrain,:);
    Ytrain = Y(1:nTrain);
    Xtest  = X(nTrain+1:end,:);
    Ytest  = Y(nTrain+1:end);
    
    %%% Initialize the swarm %%%
    positions = repmat(lb, SwarmSize, 1) + repmat((ub-lb), SwarmSize, 1).*rand(SwarmSize, dim);
    velocities = zeros(SwarmSize, dim);
    pbest = positions;
    pbestFitness = inf(SwarmSize,1);
    
    % Evaluate initial fitness
    for i = 1:SwarmSize
        pbestFitness(i) = regressionFitness(positions(i,:), Xtrain, Ytrain, Xtest, Ytest);
    end
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx,:);
    
    %%% PSO Parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% Firefly Algorithm (FA) Parameters %%%
    beta0 = 1;      % base attractiveness
    gamma = 1;      % light absorption coefficient
    alphaFA = 0.2;  % randomization parameter for FA
    
    %%% Genetic Algorithm (GA) Parameters %%%
    crossoverRate = 0.7;
    mutationRate  = 0.1;
    mutationScale = 0.1;  % scale for mutation noise
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        w_inertia = wMax - (wMax - wMin)*iter/MaxIterations;
        
        for i = 1:SwarmSize
            % PSO update:
            r1 = rand(1,dim);
            r2 = rand(1,dim);
            velocities(i,:) = w_inertia * velocities(i,:) + ...
                c1 * r1 .* (pbest(i,:) - positions(i,:)) + ...
                c2 * r2 .* (gbest - positions(i,:));
            pos_pso = positions(i,:) + velocities(i,:);
            
            % Firefly update:
            distance = norm(positions(i,:) - gbest);
            beta = beta0 * exp(-gamma * distance^2);
            pos_fa = positions(i,:) + beta * (gbest - positions(i,:)) + ...
                alphaFA * (rand(1,dim)-0.5);
            
            % Combine PSO and FA updates:
            newPos = (pos_pso + pos_fa)/2;
            
            % Enforce bounds:
            newPos = max(newPos, lb);
            newPos = min(newPos, ub);
            positions(i,:) = newPos;
            
            % Evaluate fitness:
            currentFitness = regressionFitness(newPos, Xtrain, Ytrain, Xtest, Ytest);
            
            % Update personal best:
            if currentFitness < pbestFitness(i)
                pbest(i,:) = newPos;
                pbestFitness(i) = currentFitness;
            end
            
            % Update global best:
            if currentFitness < gbestFitness
                gbest = newPos;
                gbestFitness = currentFitness;
            end
        end
        
        %%% GA Operators: Crossover and Mutation %%%
        % Randomly select two parents from the swarm.
        parentIdx = randperm(SwarmSize, 2);
        parent1 = positions(parentIdx(1),:);
        parent2 = positions(parentIdx(2),:);
        
        % Crossover (uniform crossover)
        offspring = parent1;
        for d = 1:dim
            if rand < crossoverRate
                offspring(d) = parent2(d);
            end
        end
        
        % Mutation (add small noise)
        mutation = mutationScale * (ub - lb) .* (rand(1,dim)-0.5);
        offspring = offspring + mutation;
        offspring = max(offspring, lb);
        offspring = min(offspring, ub);
        
        % Evaluate offspring fitness
        offspringFitness = regressionFitness(offspring, Xtrain, Ytrain, Xtest, Ytest);
        
        % Replace worst candidate in swarm if offspring is better.
        [worstFitness, worstIdx] = max(pbestFitness);
        if offspringFitness < worstFitness
            positions(worstIdx,:) = offspring;
            pbest(worstIdx,:) = offspring;
            pbestFitness(worstIdx) = offspringFitness;
            if offspringFitness < gbestFitness
                gbest = offspring;
                gbestFitness = offspringFitness;
            end
        end
        
        fprintf('Iteration %d/%d, Best MSE = %.4f\n', iter, MaxIterations, gbestFitness);
    end

    bestModel = trainSVMRegression(Xtrain, Ytrain, gbest(1), gbest(2));
    TestPredictions = predictSVMRegression(bestModel, Xtest);
    
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
    bestSol.TestTrue = Ytest;
    bestSol.TestPredictions = TestPredictions;
end

%% ----- Helper Functions for Regression -----

function mse = regressionFitness(params, Xtrain, Ytrain, Xtest, Ytest)
    C = params(1);
    epsilon = params(2);
    model = trainSVMRegression(Xtrain, Ytrain, C, epsilon);
    Ypred = predictSVMRegression(model, Xtest);
    mse = mean((Ytest - Ypred).^2);
end

function model = trainSVMRegression(X, Y, C, epsilon)
    [n, d] = size(X);
    w = zeros(d,1);
    b = 0;
    lr = 1e-3;
    numEpochs = 100;
    for epoch = 1:numEpochs
        grad_w = zeros(d,1);
        grad_b = 0;
        for i = 1:n
            xi = X(i,:)';
            error = Y(i) - (w' * xi + b);
            if abs(error) > epsilon
                grad_loss = -sign(error);
            else
                grad_loss = 0;
            end
            grad_w = grad_w + C * grad_loss * xi;
            grad_b = grad_b + C * grad_loss;
        end
        grad_w = grad_w + w;  % Regularization term
        w = w - lr * grad_w;
        b = b - lr * grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMRegression(model, X)
    Ypred = X * model.w + model.b;
end
