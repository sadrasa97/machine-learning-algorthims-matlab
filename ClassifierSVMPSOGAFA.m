function bestSol = ClassifierSVMPSOGAFA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)

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
    positions = lb + (ub-lb)*rand(SwarmSize, dim);
    velocities = zeros(SwarmSize, dim);
    pbest = positions;
    pbestFitness = inf(SwarmSize,1);
    
    % Evaluate initial fitness
    for i = 1:SwarmSize
        pbestFitness(i) = classificationFitness(positions(i), Xtrain, Ytrain, Xtest, Ytest);
    end
    [gbestFitness, bestIdx] = min(pbestFitness);
    gbest = pbest(bestIdx);
    
    %%% PSO Parameters %%%
    wMax = 0.9; wMin = 0.4;
    c1 = 2; c2 = 2;
    
    %%% Firefly Algorithm (FA) Parameters %%%
    beta0 = 1;
    gamma = 1;
    alphaFA = 0.2;
    
    %%% Genetic Algorithm (GA) Parameters %%%
    crossoverRate = 0.7;
    mutationRate  = 0.1;
    mutationScale = 0.1;
    
    %%% Main optimization loop %%%
    for iter = 1:MaxIterations
        w_inertia = wMax - (wMax - wMin)*iter/MaxIterations;
        for i = 1:SwarmSize
            % PSO update:
            r1 = rand;
            r2 = rand;
            velocities(i) = w_inertia * velocities(i) + ...
                c1 * r1 * (pbest(i) - positions(i)) + ...
                c2 * r2 * (gbest - positions(i));
            pos_pso = positions(i) + velocities(i);
            
            % Firefly update:
            distance = abs(positions(i) - gbest);
            beta = beta0 * exp(-gamma * distance^2);
            pos_fa = positions(i) + beta * (gbest - positions(i)) + ...
                alphaFA * (rand - 0.5);
            
            % Combine updates:
            newPos = (pos_pso + pos_fa)/2;
            newPos = max(newPos, lb);
            newPos = min(newPos, ub);
            positions(i) = newPos;
            
            % Evaluate fitness:
            currentFitness = classificationFitness(newPos, Xtrain, Ytrain, Xtest, Ytest);
            if currentFitness < pbestFitness(i)
                pbest(i) = newPos;
                pbestFitness(i) = currentFitness;
            end
            if currentFitness < gbestFitness
                gbest = newPos;
                gbestFitness = currentFitness;
            end
        end
        
        parentIdx = randperm(SwarmSize, 2);
        parent1 = positions(parentIdx(1));
        parent2 = positions(parentIdx(2));
        offspring = parent1;
        if rand < crossoverRate
            offspring = parent2;
        end
        offspring = offspring + mutationScale*(ub-lb)*(rand-0.5);
        offspring = max(offspring, lb);
        offspring = min(offspring, ub);
        offspringFitness = classificationFitness(offspring, Xtrain, Ytrain, Xtest, Ytest);
        
        [worstFitness, worstIdx] = max(pbestFitness);
        if offspringFitness < worstFitness
            positions(worstIdx) = offspring;
            pbest(worstIdx) = offspring;
            pbestFitness(worstIdx) = offspringFitness;
            if offspringFitness < gbestFitness
                gbest = offspring;
                gbestFitness = offspringFitness;
            end
        end
        
        fprintf('Iteration %d/%d, Best Misclassification Rate = %.4f\n', iter, MaxIterations, gbestFitness);
    end

    bestModel = trainSVMClassifier(Xtrain, Ytrain, gbest);
    TestPredictions = predictSVMClassifier(bestModel, Xtest);
    
    bestSol.params = gbest;
    bestSol.fitness = gbestFitness;
    bestSol.model = bestModel;
    bestSol.TestTrue = Ytest;
    bestSol.TestPredictions = TestPredictions;
end

%% ----- Helper Functions for Classification -----

function err = classificationFitness(C, Xtrain, Ytrain, Xtest, Ytest)
    model = trainSVMClassifier(Xtrain, Ytrain, C);
    Ypred = predictSVMClassifier(model, Xtest);
    err = mean(Ypred ~= Ytest);
end

function model = trainSVMClassifier(X, Y, C)
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
            margin = Y(i)*(w'*xi + b);
            if margin < 1
                grad_loss = -Y(i);
            else
                grad_loss = 0;
            end
            grad_w = grad_w + C*grad_loss*xi;
            grad_b = grad_b + C*grad_loss;
        end
        grad_w = grad_w + w;  
        w = w - lr*grad_w;
        b = b - lr*grad_b;
    end
    model.w = w;
    model.b = b;
end

function Ypred = predictSVMClassifier(model, X)
    scores = X*model.w + model.b;
    Ypred = ones(size(scores));
    Ypred(scores < 0) = -1;
end
