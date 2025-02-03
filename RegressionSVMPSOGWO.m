function bestSol = RegressionSVMPSOGWO(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize)
% RegressionSVMPSOGWO searches for optimal hyperparameters for a simple
% linear SVM regression model using a hybrid PSO-GWO algorithm.
%
%   Inputs:
%     NaturalFrequancy - feature vector (or matrix; each row a sample)
%     DamageRatio      - target values (column vector)
%     TrainRatio       - fraction of data used for training (e.g., 0.7)
%     TestRatio        - fraction of data used for testing (e.g., 0.3)
%     MaxIterations    - maximum number of iterations for the optimizer
%     SwarmSize        - number of candidate solutions in the swarm
%
%   Output:
%     bestSol - structure with the best found hyperparameters and performance
%         bestSol.params = [C, epsilon]
%         bestSol.fitness = test mean squared error (MSE)
%         bestSol.model   = structure with trained SVM parameters (w, b)

    %%% Define the decision variables (hyperparameters) %%%
    % For regression SVM we optimize:
    %   x(1) = C (regularization parameter)
    %   x(2) = epsilon (insensitive loss threshold)
    dim = 2;
    lb = [0.1, 0.001];    % lower bounds for [C, epsilon]
    ub = [1000, 1];       % upper bounds for [C, epsilon]

    %%% Prepare the data %%%
    X = NaturalFrequancy;
    Y = DamageRatio;
    N = size(X,1);
    % Randomly shuffle data
    idx = randperm(N);
    X = X(idx,:);
    Y = Y(idx);
    
    % Split data into training and testing sets according to TrainRatio and TestRatio.
    nTrain = floor(TrainRatio * N);
    nTest  = N - nTrain;
    Xtrain = X(1:nTrain,:);
    Ytrain = Y(1:nTrain);
    Xtest  = X(nTrain+1:end,:);
    Ytest  = Y(nTrain+1:end);

    %%% Initialize the swarm %%%
    positions = repmat(lb, SwarmSize, 1) + repmat((ub-lb), SwarmSize, 1) .* rand(SwarmSize, dim);
    velocities = zeros(SwarmSize, dim);
    pbest = positions; % personal best positions
    pbestFitness = inf(SwarmSize, 1);
    
    % Evaluate initial fitness for each candidate
    for i = 1:SwarmSize
        fitness = regressionFitness(positions(i,:), Xtrain, Ytrain, Xtest, Ytest);
        pbestFitness(i) = fitness;
    end
    % Global best
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx,:);
    
    % Also keep track of the best three solutions for the GWO update:
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

    %%% PSO parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        % Linearly decreasing inertia weight
        w_inertia = wMax - (wMax - wMin)*iter/MaxIterations;
        
        for i = 1:SwarmSize
            % PSO velocity update
            r1 = rand(1,dim);
            r2 = rand(1,dim);
            velocities(i,:) = w_inertia * velocities(i,:) ...
                + c1 * r1 .* (pbest(i,:) - positions(i,:)) ...
                + c2 * r2 .* (gbest - positions(i,:));
            pos_pso = positions(i,:) + velocities(i,:);
            
            % GWO update:
            % Compute distances from the top three solutions:
            A1 = 2*rand(1,dim) - 1;
            A2 = 2*rand(1,dim) - 1;
            A3 = 2*rand(1,dim) - 1;
            D_alpha = abs(A1 .* (alpha - positions(i,:)));
            D_beta  = abs(A2 .* (beta  - positions(i,:)));
            D_delta = abs(A3 .* (delta - positions(i,:)));
            X1 = alpha - D_alpha;
            X2 = beta  - D_beta;
            X3 = delta - D_delta;
            pos_gwo = (X1 + X2 + X3) / 3;
            
            % Combine the two updates (simple average)
            newPos = (pos_pso + pos_gwo) / 2;
            
            % Keep within bounds
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
        
        % Update the top three solutions for GWO:
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
        
        % (Optional) display progress:
        fprintf('Iteration %d/%d, Best MSE = %.4f\n', iter, MaxIterations, gbestFitness);
    end

    % Once the best hyperparameters have been found, retrain the SVM on all training data.
    bestModel = trainSVMRegression(Xtrain, Ytrain, gbest(1), gbest(2));
    
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
end

%% ----------------- Helper Functions for Regression -----------------

function mse = regressionFitness(params, Xtrain, Ytrain, Xtest, Ytest)
    % params: [C, epsilon]
    C = params(1);
    epsilon = params(2);
    % Train a simple linear SVM regression model
    model = trainSVMRegression(Xtrain, Ytrain, C, epsilon);
    % Predict on test data
    Ypred = predictSVMRegression(model, Xtest);
    % Use mean squared error as fitness (lower is better)
    mse = mean((Ytest - Ypred).^2);
end

function model = trainSVMRegression(X, Y, C, epsilon)
    % A simple gradient-descent training for a linear SVM regression.
    % Model: f(x) = w'*x + b.
    % Loss: regularization (0.5*||w||^2) + C * sum_i L_epsilon(error)
    % where L_epsilon(error) = max(0, |error| - epsilon)
    
    [n, d] = size(X);
    % Initialize weights and bias
    w = zeros(d,1);
    b = 0;
    
    lr = 1e-3;         % learning rate
    numEpochs = 100;   % number of epochs (for demonstration)
    
    for epoch = 1:numEpochs
        grad_w = zeros(d,1);
        grad_b = 0;
        for i = 1:n
            xi = X(i,:)';
            error = Y(i) - (w' * xi + b);
            % Compute subgradient for epsilon-insensitive loss
            if abs(error) > epsilon
                grad_loss = -sign(error);
            else
                grad_loss = 0;
            end
            grad_w = grad_w + C * grad_loss * xi;
            grad_b = grad_b + C * grad_loss;
        end
        % Add gradient of regularization term
        grad_w = grad_w + w;
        
        % Update parameters
        w = w - lr * grad_w;
        b = b - lr * grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMRegression(model, X)
    Ypred = X * model.w + model.b;
end
