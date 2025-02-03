function bestSol = ClassifierSVMPSOGWO(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
% ClassifierSVMPSOGWO searches for an optimal hyperparameter (C) for a
% simple linear SVM classifier using a hybrid PSO-GWO algorithm.
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
%     bestSol - structure with the best found hyperparameter and performance
%         bestSol.params = C
%         bestSol.fitness = test misclassification rate
%         bestSol.model   = structure with trained SVM parameters (w, b)

    %%% Here we optimize only one decision variable: C (regularization parameter) %%%
    dim = 1;
    lb = 0.1;
    ub = 1000;
    
    %%% Prepare the data %%%
    X = NaturalFrequancy;
    Y = DamageLocation;
    % (If symmetry-related processing is needed, you could add it here.)
    N = size(X,1);
    idx = randperm(N);
    X = X(idx,:);
    Y = Y(idx);
    
    % Split data into training and testing sets.
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
    
    % Evaluate initial fitness for each candidate
    for i = 1:SwarmSize
        fitness = classificationFitness(positions(i), Xtrain, Ytrain, Xtest, Ytest);
        pbestFitness(i) = fitness;
    end
    % Global best
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx);
    
    % Keep best three solutions for GWO (for dim=1, these are scalars)
    [sortedFitness, sortedIdx] = sort(pbestFitness);
    alpha = positions(sortedIdx(1));
    if SwarmSize>=2
        beta = positions(sortedIdx(2));
    else
        beta = alpha;
    end
    if SwarmSize>=3
        delta = positions(sortedIdx(3));
    else
        delta = beta;
    end

    %%% PSO parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        w_inertia = wMax - (wMax - wMin)*iter/MaxIterations;
        for i = 1:SwarmSize
            % PSO update
            r1 = rand;
            r2 = rand;
            velocities(i) = w_inertia * velocities(i) + c1 * r1 * (pbest(i) - positions(i)) + c2 * r2 * (gbest - positions(i));
            pos_pso = positions(i) + velocities(i);
            
            % GWO update:
            A1 = 2*rand - 1;
            A2 = 2*rand - 1;
            A3 = 2*rand - 1;
            D_alpha = abs(A1*(alpha - positions(i)));
            D_beta  = abs(A2*(beta  - positions(i)));
            D_delta = abs(A3*(delta - positions(i)));
            X1 = alpha - D_alpha;
            X2 = beta - D_beta;
            X3 = delta - D_delta;
            pos_gwo = (X1 + X2 + X3)/3;
            
            % Combine the two updates
            newPos = (pos_pso + pos_gwo)/2;
            % Apply bounds
            newPos = max(newPos, lb);
            newPos = min(newPos, ub);
            
            positions(i) = newPos;
            % Evaluate new fitness
            fitness = classificationFitness(newPos, Xtrain, Ytrain, Xtest, Ytest);
            % Update personal best if improved
            if fitness < pbestFitness(i)
                pbest(i) = newPos;
                pbestFitness(i) = fitness;
            end
            % Update global best if improved
            if fitness < gbestFitness
                gbest = newPos;
                gbestFitness = fitness;
            end
        end
        
        % Update the best three solutions for GWO
        [sortedFitness, sortedIdx] = sort(pbestFitness);
        alpha = positions(sortedIdx(1));
        if SwarmSize>=2
            beta = positions(sortedIdx(2));
        else
            beta = alpha;
        end
        if SwarmSize>=3
            delta = positions(sortedIdx(3));
        else
            delta = beta;
        end
        
        fprintf('Iteration %d/%d, Best Misclass Rate = %.4f\n', iter, MaxIterations, gbestFitness);
    end
    
    % Retrain the SVM classifier on the full training data using the best found C.
    bestModel = trainSVMClassifier(Xtrain, Ytrain, gbest);
    
    % Compute predictions on the test set
    TestPredictions = predictSVMClassifier(bestModel, Xtest);
    
    % Add test true labels and predictions to the output structure
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
    bestSol.TestTrue = Ytest;
    bestSol.TestPredictions = TestPredictions;
end


%% ----------------- Helper Functions for Classification -----------------

function err = classificationFitness(C, Xtrain, Ytrain, Xtest, Ytest)
    % Train a simple linear SVM classifier using parameter C.
    model = trainSVMClassifier(Xtrain, Ytrain, C);
    % Predict on test set
    Ypred = predictSVMClassifier(model, Xtest);
    % Compute misclassification rate
    err = mean(Ypred ~= Ytest);
end

function model = trainSVMClassifier(X, Y, C)
    % A simple gradient-descent training for a linear SVM classifier.
    % Model: f(x) = w'*x + b.
    % Loss: regularization (0.5*||w||^2) + C * sum_i max(0, 1 - y*(w'*x+b))
    
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
        grad_w = grad_w + w; % add regularization gradient
        w = w - lr * grad_w;
        b = b - lr * grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMClassifier(model, X)
    scores = X * model.w + model.b;
    % Predict class labels: assume binary classification with labels +1 and -1.
    Ypred = ones(size(scores));
    Ypred(scores < 0) = -1;
end
