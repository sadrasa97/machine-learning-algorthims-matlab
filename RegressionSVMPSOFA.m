function bestSol = RegressionSVMPSOFA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
% RegressionSVMPSOFA searches for optimal hyperparameters for a simple
% linear SVM regression model using a hybrid PSO–Firefly Algorithm (FA).
%
%   Inputs:
%     NaturalFrequancy - feature matrix (each row a sample)
%     DamageRatio      - target values (column vector)
%     TrainRatio       - fraction of data used for training (e.g., 0.7)
%     TestRatio        - fraction of data used for testing (e.g., 0.3)
%     MaxIterations    - maximum number of iterations for the optimizer
%     SwarmSize        - number of candidate solutions in the swarm
%
%   Output:
%     bestSol - structure with the best found hyperparameters, performance, 
%               and model. Fields:
%         params          = [C, epsilon]
%         fitness         = test mean squared error (MSE)
%         model           = trained SVM regression model (w, b)
%         TestTrue        = true test labels
%         TestPredictions = predictions on the test set

    %%% Define decision variables %%%
    % For regression SVM, we optimize:
    %   x(1) = C (regularization parameter)
    %   x(2) = epsilon (epsilon-insensitive zone)
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
    positions = repmat(lb, SwarmSize, 1) + repmat((ub-lb), SwarmSize, 1) .* rand(SwarmSize, dim);
    velocities = zeros(SwarmSize, dim);
    pbest = positions;
    pbestFitness = inf(SwarmSize, 1);
    
    % Evaluate initial fitness for each candidate
    for i = 1:SwarmSize
        fitness = regressionFitness(positions(i,:), Xtrain, Ytrain, Xtest, Ytest);
        pbestFitness(i) = fitness;
    end
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx,:);
    
    %%% PSO Parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% Firefly Algorithm Parameters %%%
    beta0 = 1;     % base attractiveness
    gamma = 1;     % light absorption coefficient
    alpha = 0.2;   % randomization parameter
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        w_inertia = wMax - (wMax - wMin) * iter / MaxIterations;
        
        for i = 1:SwarmSize
            % PSO update:
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            velocities(i,:) = w_inertia * velocities(i,:) + ...
                c1 * r1 .* (pbest(i,:) - positions(i,:)) + ...
                c2 * r2 .* (gbest - positions(i,:));
            pos_pso = positions(i,:) + velocities(i,:);
            
            % Firefly update:
            % In the FA, a candidate moves toward a brighter firefly. Here we use
            % the global best (gbest) as the attractor.
            distance = norm(positions(i,:) - gbest);
            beta = beta0 * exp(-gamma * distance^2);
            pos_fa = positions(i,:) + beta * (gbest - positions(i,:)) + ...
                alpha * (rand(1, dim)-0.5);
            
            % Combine the two updates (simple average)
            newPos = (pos_pso + pos_fa) / 2;
            
            % Enforce bounds
            newPos = max(newPos, lb);
            newPos = min(newPos, ub);
            
            positions(i,:) = newPos;
            
            % Evaluate new fitness
            fitness = regressionFitness(newPos, Xtrain, Ytrain, Xtest, Ytest);
            
            % Update personal best if improved
            if fitness < pbestFitness(i)
                pbest(i,:) = newPos;
                pbestFitness(i) = fitness;
            end
            
            % Update global best if improved
            if fitness < gbestFitness
                gbest = newPos;
                gbestFitness = fitness;
            end
        end
        
        % (Optional) Display progress:
        fprintf('Iteration %d/%d, Best MSE = %.4f\n', iter, MaxIterations, gbestFitness);
    end

    % Retrain the SVM regression model on training data using best hyperparameters.
    bestModel = trainSVMRegression(Xtrain, Ytrain, gbest(1), gbest(2));
    TestPredictions = predictSVMRegression(bestModel, Xtest);
    
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
    bestSol.TestTrue = Ytest;
    bestSol.TestPredictions = TestPredictions;
end

%% --------- Helper Functions for Regression ---------

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
        grad_w = grad_w + w; % Regularization gradient
        w = w - lr * grad_w;
        b = b - lr * grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMRegression(model, X)
    Ypred = X * model.w + model.b;
end


