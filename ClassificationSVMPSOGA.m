function ClassificationSol = ClassificationSVMPSOGA(...
    NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, ...
    MaxIterations, SwarmSize, PopulationSize, MaxGen ...
)
% SVMPSOGA_Classification optimizes SVM model using PSO and custom GA with LIBSVM.
%
% INPUTS:
%   NaturalFrequency - Feature matrix (n x d).
%   DamageLocation   - Class labels (n x 1).
%   TrainRatio       - Training data ratio.
%   TestRatio        - Test data ratio.
%   MaxIterations    - Max PSO iterations.
%   SwarmSize        - Number of PSO particles.
%   PopulationSize   - Number of GA individuals.
%   MaxGen           - Max GA generations.
%
% OUTPUT:
%   ClassificationSol - Struct with optimized SVM model and parameters.

    % Data Splitting
    n = size(NaturalFrequancy, 1);
    cv1 = cvpartition(n, 'HoldOut', 1 - TrainRatio);
    XTrain = NaturalFrequency(training(cv1), :);
    yTrain = DamageLocation(training(cv1));
    XTest = NaturalFrequency(test(cv1), :);
    yTest = DamageLocation(test(cv1));

    % PSO Optimization for Hyperparameters
    lb = [0.1, 0.001];  % Lower bounds: [C, gamma]
    ub = [100, 10];     % Upper bounds
    [~, ~] = PSO(@(p) SVM_Cost(p, XTrain, yTrain), SwarmSize, lb, ub, MaxIterations);

    % GA Optimization without Toolbox
    bestGA = customGA(@(p) SVM_Cost(p, XTrain, yTrain), lb, ub, PopulationSize, MaxGen, 'iter');

    % Train Final SVM Model with LIBSVM
    best_C = bestGA(1);
    best_gamma = bestGA(2);
    model = svmtrain(yTrain, XTrain, sprintf('-s 0 -t 2 -c %f -g %f', best_C, best_gamma));

    % Predictions and Accuracy
    [yPredTest, accuracy, ~] = svmpredict(yTest, XTest, model);

    % Return Results
    ClassificationSol.Model = model;
    ClassificationSol.BestParams = struct('C', best_C, 'Gamma', best_gamma);
    ClassificationSol.TestAccuracy = accuracy(1);
end

function error = SVM_Cost(params, X, y)
    try
        model = svmtrain(y, X, sprintf('-s 0 -t 2 -c %f -g %f -v 5', params(1), params(2)));
        error = 100 - model;  % Minimize error (maximize accuracy)
    catch
        error = inf;
    end
end

function [bestPosition, bestCost] = PSO(costFunc, swarmSize, lb, ub, maxIter)
    dim = length(lb);
    pos = repmat(lb, swarmSize, 1) + rand(swarmSize, dim) .* repmat(ub - lb, swarmSize, 1);
    vel = zeros(swarmSize, dim);
    pbest = pos;
    pbestCost = arrayfun(@(i) costFunc(pos(i, :)), 1:swarmSize)';
    [gbestCost, idx] = min(pbestCost);
    gbest = pos(idx, :);
    
    w = 0.9; w_min = 0.4; c1 = 2; c2 = 2;
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