function bestSol = ClassifierSVMPSOFA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
% ClassifierSVMPSOFA searches for an optimal hyperparameter (C) for a simple
% linear SVM classifier using a hybrid PSO–Firefly Algorithm.
%
%   Inputs:
%     NaturalFrequancy - feature matrix (each row a sample)
%     DamageLocation   - class labels (assumed to be +1 or -1)
%     TrainRatio       - fraction of data used for training (e.g., 0.7)
%     TestRatio        - fraction of data used for testing (e.g., 0.3)
%     MaxIterations    - maximum number of iterations for the optimizer
%     SwarmSize        - number of candidate solutions in the swarm
%     isSymmetry, Symmetry - additional parameters (unused in this demo)
%
%   Output:
%     bestSol - structure with the best found hyperparameter, performance,
%               and model. Fields:
%         params          = C
%         fitness         = test misclassification rate
%         model           = trained SVM classifier (w, b)
%         TestTrue        = true test labels
%         TestPredictions = predictions on the test set

    %%% Decision variable: only C for classification %%%
    dim = 1;
    lb = 0.1;
    ub = 1000;
    
    %%% Prepare the data %%%
    X = NaturalFrequancy;
    Y = DamageLocation;
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
    positions = lb + (ub - lb)*rand(SwarmSize, dim);
    velocities = zeros(SwarmSize, dim);
    pbest = positions;
    pbestFitness = inf(SwarmSize, 1);
    
    for i = 1:SwarmSize
        fitness = classificationFitness(positions(i), Xtrain, Ytrain, Xtest, Ytest);
        pbestFitness(i) = fitness;
    end
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx);
    
    %%% PSO Parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% Firefly Algorithm Parameters %%%
    beta0 = 1;
    gamma = 1;
    alpha = 0.2;
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        w_inertia = wMax - (wMax - wMin) * iter / MaxIterations;
        for i = 1:SwarmSize
            % PSO update:
            r1 = rand;
            r2 = rand;
            velocities(i) = w_inertia * velocities(i) + ...
                c1 * r1 * (pbest(i) - positions(i)) + ...
                c2 * r2 * (gbest - positions(i));
            pos_pso = positions(i) + velocities(i);
            
            % Firefly update (using global best as attractor):
            distance = abs(positions(i) - gbest);
            beta = beta0 * exp(-gamma * distance^2);
            pos_fa = positions(i) + beta * (gbest - positions(i)) + ...
                alpha * (rand - 0.5);
            
            % Combine the two updates
            newPos = (pos_pso + pos_fa) / 2;
            
            % Enforce bounds
            newPos = max(newPos, lb);
            newPos = min(newPos, ub);
            
            positions(i) = newPos;
            
            % Evaluate fitness
            fitness = classificationFitness(newPos, Xtrain, Ytrain, Xtest, Ytest);
            if fitness < pbestFitness(i)
                pbest(i) = newPos;
                pbestFitness(i) = fitness;
            end
            if fitness < gbestFitness
                gbest = newPos;
                gbestFitness = fitness;
            end
        end
        
        fprintf('Iteration %d/%d, Best Misclass Rate = %.4f\n', iter, MaxIterations, gbestFitness);
    end
    
    % Retrain the SVM classifier using the best found parameter C.
    bestModel = trainSVMClassifier(Xtrain, Ytrain, gbest);
    TestPredictions = predictSVMClassifier(bestModel, Xtest);
    
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
    bestSol.TestTrue = Ytest;
    bestSol.TestPredictions = TestPredictions;
end

%% --------- Helper Functions for Classification ---------

function err = classificationFitness(C, Xtrain, Ytrain, Xtest, Ytest)
    model = trainSVMClassifier(Xtrain, Ytrain, C);
    Ypred = predictSVMClassifier(model, Xtest);
    err = mean(Ypred ~= Ytest);
end

function model = trainSVMClassifier(X, Y, C)
    % Train a simple linear SVM classifier using gradient descent.
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
            margin = Y(i) * (w' * xi + b);
            if margin < 1
                grad_loss = -Y(i);
            else
                grad_loss = 0;
            end
            grad_w = grad_w + C * grad_loss * xi;
            grad_b = grad_b + C * grad_loss;
        end
        grad_w = grad_w + w;  % Regularization
        w = w - lr * grad_w;
        b = b - lr * grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMClassifier(model, X)
    scores = X * model.w + model.b;
    Ypred = ones(size(scores));
    Ypred(scores < 0) = -1;
end
