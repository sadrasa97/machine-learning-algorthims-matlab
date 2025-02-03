function RegressionSol = RegressionANNPSO(...
    NaturalFrequency, DamageRatio, TrainRatio, TestRatio, ...
    MaxIterations, SwarmSize ...
)
    %% Data Splitting
    n = size(NaturalFrequency,1);
    cv1 = cvpartition(n, 'HoldOut', 1 - TrainRatio);
    XTrain = NaturalFrequency(training(cv1),:);
    yTrain = DamageRatio(training(cv1));
    XRemain = NaturalFrequency(test(cv1),:);
    yRemain = DamageRatio(test(cv1));
    
    cv2 = cvpartition(size(XRemain,1), 'HoldOut', TestRatio/(1-TrainRatio));
    XTest = XRemain(test(cv2),:);
    yTest = yRemain(test(cv2));
    
    %% PSO Setup
    dim = 2;  % [Number of Neurons, Learning Rate]
    lb = [5, 0.0001]; % Lower bounds
    ub = [100, 0.1];  % Upper bounds
    
    % Initialize swarm
    pos = lb + rand(SwarmSize, dim) .* (ub - lb);
    vel = zeros(SwarmSize, dim);
    pbest = pos;
    pbestCost = inf(SwarmSize, 1);
    
    % Evaluate cost for each particle
    for i = 1:SwarmSize
        pbestCost(i) = costFunctionRegression(pos(i,:), XTrain, yTrain);
    end
    
    [gbestCost, idx] = min(pbestCost);
    gbest = pos(idx,:);
    
    %% PSO Optimization Loop
    w = 0.9; w_min = 0.4; c1 = 2; c2 = 2;
    
    for iter = 1:MaxIterations
        w = w - ((0.9 - w_min) / MaxIterations);
        for i = 1:SwarmSize
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            vel(i,:) = w * vel(i,:) + c1 * r1 .* (pbest(i,:) - pos(i,:)) + c2 * r2 .* (gbest - pos(i,:));
            pos(i,:) = max(min(pos(i,:) + vel(i,:), ub), lb);
            
            cost = costFunctionRegression(pos(i,:), XTrain, yTrain);
            if cost < pbestCost(i)
                pbest(i,:) = pos(i,:);
                pbestCost(i) = cost;
            end
            if cost < gbestCost
                gbest = pos(i,:);
                gbestCost = cost;
            end
        end
        fprintf('Iteration %d: Best MSE = %f\n', iter, gbestCost);
    end

    %% Train Final ANN Model
    best_neurons = round(gbest(1));
    best_lr = gbest(2);
    
    net = feedforwardnet(best_neurons);
    net.trainParam.lr = best_lr;
    net.trainParam.epochs = 100;
    net = train(net, XTrain', yTrain');
    
    yPredTrain = net(XTrain')';
    yPredTest = net(XTest')';
    
    trainMSE = mean((yTrain - yPredTrain).^2);
    testMSE = mean((yTest - yPredTest).^2);
    
    %% Return Results
    RegressionSol.Model = net;
    RegressionSol.BestParams = struct('Neurons', best_neurons, 'LearningRate', best_lr);
    RegressionSol.TrainMSE = trainMSE;
    RegressionSol.TestMSE = testMSE;
    RegressionSol.gbestCost = gbestCost;

    %% Nested Cost Function
    function err = costFunctionRegression(params, X_cv, y_cv)
        neurons = round(params(1));
        lr = params(2);
        
        try
            net = feedforwardnet(neurons);
            net.trainParam.lr = lr;
            net.trainParam.epochs = 50;
            net = train(net, X_cv', y_cv');
            yPred = net(X_cv')';
            err = mean((y_cv - yPred).^2);
        catch
            err = inf;
        end
    end
end


