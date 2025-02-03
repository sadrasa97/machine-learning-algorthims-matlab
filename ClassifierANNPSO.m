function ClassificationSol = ClassifierANNPSO(...
    NaturalFrequency, DamageLocation, TrainRatio, TestRatio, ...
    MaxIterations, SwarmSize, isSymmetry, Symmetry ...
)
    %% Data Splitting
    n = size(NaturalFrequency,1);
    cv1 = cvpartition(n, 'HoldOut', 1 - TrainRatio);
    XTrain = NaturalFrequency(training(cv1),:);
    yTrain = DamageLocation(training(cv1));
    XRemain = NaturalFrequency(test(cv1),:);
    yRemain = DamageLocation(test(cv1));
    
    cv2 = cvpartition(size(XRemain,1), 'HoldOut', TestRatio/(1-TrainRatio));
    XTest = XRemain(test(cv2),:);
    yTest = yRemain(test(cv2));

    %% PSO Setup
    dim = 2;  
    lb = [5, 0.0001]; 
    ub = [100, 0.1];
    
    pos = lb + rand(SwarmSize, dim) .* (ub - lb);
    vel = zeros(SwarmSize, dim);
    pbest = pos;
    pbestCost = inf(SwarmSize, 1);
    
    for i = 1:SwarmSize
        pbestCost(i) = costFunctionClassification(pos(i,:), XTrain, yTrain);
    end
    
    [gbestCost, idx] = min(pbestCost);
    gbest = pos(idx,:);

    %% Train Final ANN Model
    best_neurons = round(gbest(1));
    best_lr = gbest(2);
    
    net = patternnet(best_neurons);
    net.trainParam.lr = best_lr;
    net = train(net, XTrain', yTrain');
    
    yPredTest = net(XTest') > 0.5;
    testAcc = sum(yPredTest' == yTest) / numel(yTest);

    ClassificationSol.Model = net;
    ClassificationSol.TestAccuracy = testAcc;
end
