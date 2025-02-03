function bestSol = RegressionSVMPSOGAGWO(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)

    dim = 2;
    lb = [0.1, 0.001];    % lower bounds for [C, epsilon]
    ub = [1000, 1];       % upper bounds for [C, epsilon]

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
    pbestFitness = inf(SwarmSize, 1);
    
    % Evaluate initial fitness for each candidate
    for i = 1:SwarmSize
        pbestFitness(i) = regressionFitness(positions(i,:), Xtrain, Ytrain, Xtest, Ytest);
    end
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx,:);
    
    %%% GWO initialization: Determine alpha, beta, delta (best three solutions)
    [sortedFitness, sortedIdx] = sort(pbestFitness);
    alpha = positions(sortedIdx(1),:);
    if SwarmSize>=2
        beta = positions(sortedIdx(2),:);
    else
        beta = alpha;
    end
    if SwarmSize>=3
        delta = positions(sortedIdx(3),:);
    else
        delta = beta;
    end

    %%% PSO Parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% GA Parameters %%
    crossoverRate = 0.7;
    mutationRate  = 0.1;
    mutationScale = 0.1;  % relative scale for mutation
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        % Linearly decrease inertia weight
        w_inertia = wMax - (wMax - wMin)*iter/MaxIterations;
        
        for i = 1:SwarmSize
            % PSO update:
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            velocities(i,:) = w_inertia * velocities(i,:) + ...
                c1 * r1 .* (pbest(i,:) - positions(i,:)) + ...
                c2 * r2 .* (gbest - positions(i,:));
            pos_pso = positions(i,:) + velocities(i,:);
            
            % GWO update:
            % Generate random coefficients for each best solution
            A1 = 2*rand(1,dim) - 1;
            A2 = 2*rand(1,dim) - 1;
            A3 = 2*rand(1,dim) - 1;
            D_alpha = abs(A1 .* (alpha - positions(i,:)));
            D_beta  = abs(A2 .* (beta - positions(i,:)));
            D_delta = abs(A3 .* (delta - positions(i,:)));
            X1 = alpha - D_alpha;
            X2 = beta - D_beta;
            X3 = delta - D_delta;
            pos_gwo = (X1 + X2 + X3)/3;
            
            % GA update:
            % Randomly select two parents from the swarm
            parentIdx = randperm(SwarmSize,2);
            parent1 = positions(parentIdx(1),:);
            parent2 = positions(parentIdx(2),:);
            offspring = parent1;
            for d = 1:dim
                if rand < crossoverRate
                    offspring(d) = parent2(d);
                end
            end
            % Mutation (add random noise)
            mutation = mutationScale * (ub - lb) .* (rand(1, dim)-0.5);
            offspring = offspring + mutation;
            offspring = max(offspring, lb);
            offspring = min(offspring, ub);
            pos_ga = offspring;
            
            % Combine the three updates (average them)
            newPos = (pos_pso + pos_gwo + pos_ga)/3;
            
            % Enforce boundaries
            newPos = max(newPos, lb);
            newPos = min(newPos, ub);
            positions(i,:) = newPos;
            
            % Evaluate fitness of new candidate
            currentFitness = regressionFitness(newPos, Xtrain, Ytrain, Xtest, Ytest);
            
            % Update personal best if improved
            if currentFitness < pbestFitness(i)
                pbest(i,:) = newPos;
                pbestFitness(i) = currentFitness;
            end
            
            % Update global best if improved
            if currentFitness < gbestFitness
                gbest = newPos;
                gbestFitness = currentFitness;
            end
        end
        
        % Update GWO best three: alpha, beta, delta
        [sortedFitness, sortedIdx] = sort(pbestFitness);
        alpha = positions(sortedIdx(1),:);
        if SwarmSize>=2
            beta = positions(sortedIdx(2),:);
        else
            beta = alpha;
        end
        if SwarmSize>=3
            delta = positions(sortedIdx(3),:);
        else
            delta = beta;
        end
        
        fprintf('Iteration %d/%d, Best MSE = %.4f\n', iter, MaxIterations, gbestFitness);
    end

    % Retrain the SVM regression model on training data using best parameters.
    bestModel = trainSVMRegression(Xtrain, Ytrain, gbest(1), gbest(2));
    TestPredictions = predictSVMRegression(bestModel, Xtest);
    
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
    bestSol.TestTrue = Ytest;
    bestSol.TestPredictions = TestPredictions;
end

%% ---- Helper Functions for Regression ----

function mse = regressionFitness(params, Xtrain, Ytrain, Xtest, Ytest)
    C = params(1);
    epsilon = params(2);
    model = trainSVMRegression(Xtrain, Ytrain, C, epsilon);
    Ypred = predictSVMRegression(model, Xtest);
    mse = mean((Ytest - Ypred).^2);
end

function model = trainSVMRegression(X, Y, C, epsilon)
    % Train a simple linear SVM regression model using gradient descent.
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
        grad_w = grad_w + w;  % Regularization gradient
        w = w - lr * grad_w;
        b = b - lr * grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMRegression(model, X)
    Ypred = X * model.w + model.b;
end
