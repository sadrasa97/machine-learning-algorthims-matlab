function RegressionSol = RegressionSVMPSOGA(...  
    NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, ...  
    MaxIterations, SwarmSize, PopulationSize, MaxGen)

% SVMPSOGA_Regression optimizes SVR model using PSO and custom GA.
%
% INPUTS:
%   NaturalFrequency - Feature matrix (n x d).
%   DamageRatio      - Target variable (n x 1).
%   TrainRatio       - Training data ratio.
%   TestRatio        - Test data ratio.
%   MaxIterations    - Max PSO iterations.
%   SwarmSize        - Number of PSO particles.
%   PopulationSize   - Number of GA individuals.
%   MaxGen           - Max GA generations.
%
% OUTPUT:
%   RegressionSol - Struct with optimized SVR model and parameters.

    % Data Splitting (Manual)
    n = size(NaturalFrequancy, 1);
    indices = randperm(n);
    trainSize = round(n * TrainRatio);
    
    XTrain = NaturalFrequancy(indices(1:trainSize), :);
    yTrain = DamageRatio(indices(1:trainSize));

    XTest = NaturalFrequancy(indices(trainSize+1:end), :);
    yTest = DamageRatio(indices(trainSize+1:end));

    % PSO Optimization
    lb = [0.1, 0.001, 0.001]; % [C, epsilon, gamma]
    ub = [100, 1, 10];
    [~, ~] = PSO(@(p) SVR_Cost(p, XTrain, yTrain), SwarmSize, lb, ub, MaxIterations);

    % GA Optimization without Toolbox
    bestGA = customGA(@(p) SVR_Cost(p, XTrain, yTrain), lb, ub, PopulationSize, MaxGen, 'iter');

    % Train Final SVR Model
    best_C = bestGA(1);
    best_epsilon = bestGA(2);
    best_gamma = bestGA(3);
    
    % Custom SVR Training Function
    model = trainSVR(XTrain, yTrain, best_C, best_epsilon, best_gamma);

    % Predictions and MSE
    yPredTest = predictSVR(model, XTest);
    testMSE = mean((yTest - yPredTest).^2);

    % Return Results
    RegressionSol.Model = model;
    RegressionSol.BestParams = struct('C', best_C, 'Epsilon', best_epsilon, 'Gamma', best_gamma);
    RegressionSol.TestMSE = testMSE;
end

%% Custom SVR Training Function
function model = trainSVR(X, y, C, epsilon, gamma)
    % Train a simple SVR model using a Gaussian kernel
    n = size(X, 1);
    K = exp(-gamma * pdist2(X, X).^2); % Gaussian kernel
    alpha = (K + (1/C) * eye(n)) \ y; % Solve for weights
    
    model.C = C;
    model.Epsilon = epsilon;
    model.Gamma = gamma;
    model.Alpha = alpha;
    model.X = X;
end

%% SVR Prediction Function
function yPred = predictSVR(model, XTest)
    KTest = exp(-model.Gamma * pdist2(XTest, model.X).^2); % Gaussian kernel
    yPred = KTest * model.Alpha;
end

%% SVR Cost Function
function error = SVR_Cost(params, X, y)
    try
        % Example: use 5-fold cross-validation
        cv = cvpartition(size(X,1), 'KFold', 5);
        cvError = zeros(cv.NumTestSets, 1);
        for k = 1:cv.NumTestSets
            XTrain = X(cv.training(k), :);
            yTrain = y(cv.training(k));
            XVal = X(cv.test(k), :);
            yVal = y(cv.test(k));

            model = trainSVR(XTrain, yTrain, params(1), params(2), params(3));
            yPred = predictSVR(model, XVal);
            cvError(k) = mean((yVal - yPred).^2);
        end
        error = mean(cvError);
    catch
        error = inf;
    end
end

%% Particle Swarm Optimization (PSO)
function [bestPosition, bestCost] = PSO(costFunc, swarmSize, lb, ub, maxIter)
    dim = length(lb);
    pos = repmat(lb, swarmSize, 1) + rand(swarmSize, dim) .* repmat(ub - lb, swarmSize, 1);
    vel = zeros(swarmSize, dim);
    pbest = pos;
    pbestCost = arrayfun(@(i) costFunc(pos(i, :)), 1:swarmSize)';
    [gbestCost, idx] = min(pbestCost);
    gbest = pos(idx, :);
    
    w = 0.9; c1 = 2; c2 = 2;
    for iter = 1:maxIter
        w = w - (0.5 / maxIter);
        for i = 1:swarmSize
            r1 = rand(1, dim); r2 = rand(1, dim);
            vel(i, :) = w * vel(i, :) + c1*r1.*(pbest(i, :) - pos(i, :)) + c2*r2.*(gbest - pos(i, :));
            pos(i, :) = max(min(pos(i, :) + vel(i, :), ub), lb);
            currentCost = costFunc(pos(i, :));
            if currentCost < pbestCost(i)
                pbest(i, :) = pos(i, :);
                pbestCost(i) = currentCost;
                if currentCost < gbestCost
                    gbest = pos(i, :);
                    gbestCost = currentCost;
                end
            end
        end
    end
    bestPosition = gbest;
    bestCost = gbestCost;
end

%% Genetic Algorithm (GA)
function bestIndividual = customGA(costFunc, lb, ub, populationSize, maxGen, display)
    crossoverProb = 0.8;
    mutationProb = 0.1;
    elitismCount = 2;
    dim = length(lb);
    population = repmat(lb, populationSize, 1) + rand(populationSize, dim) .* repmat(ub - lb, populationSize, 1);
    
    for gen = 1:maxGen
        costs = arrayfun(@(i) costFunc(population(i,:)), 1:populationSize);
        [~, idx] = sort(costs);
        population = population(idx, :);
        
        if strcmp(display, 'iter')
            fprintf('Generation %d: Best Cost = %.4f\n', gen, costs(idx(1)));
        end
        
        newPopulation = population(1:elitismCount, :);
        for i = 1:ceil((populationSize - elitismCount)/2)
            p1 = population(randi([1, populationSize/2]), :);
            p2 = population(randi([1, populationSize/2]), :);
            
            if rand < crossoverProb
                alpha = rand(1, dim);
                c1 = alpha .* p1 + (1 - alpha) .* p2;
                c2 = alpha .* p2 + (1 - alpha) .* p1;
            else
                c1 = p1; c2 = p2;
            end
            
            for j = 1:dim
                if rand < mutationProb
                    c1(j) = lb(j) + (ub(j) - lb(j)) * rand();
                end
                if rand < mutationProb
                    c2(j) = lb(j) + (ub(j) - lb(j)) * rand();
                end
            end
            newPopulation = [newPopulation; c1; c2];
        end
        population = newPopulation(1:populationSize, :);
    end
    costs = arrayfun(@(i) costFunc(population(i,:)), 1:populationSize);
    [~, bestIdx] = min(costs);
    bestIndividual = population(bestIdx, :);
end
