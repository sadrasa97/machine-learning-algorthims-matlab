function ClassificationSol = ClassifierSVMPSO(X, y, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
% ClassifierSVMPSO performs SVM classification with hyperparameter tuning via PSO.
%
% INPUTS:
%   X             - Feature matrix (observations x features)
%   y             - Class labels (as numeric or categorical values)
%   TrainRatio    - Ratio of data used for training (e.g. 0.7)
%   TestRatio     - Ratio of data used for testing (e.g. 0.15)
%   MaxIterations - Maximum number of PSO iterations
%   SwarmSize     - Number of particles in the swarm
%   isSymmetry    - (Optional) flag for symmetry handling (not used in PSO here)
%   Symmetry      - (Optional) symmetry matrix (not used in PSO here)
%
% OUTPUT:
%   ClassificationSol - Structure containing:
%         .Model          : Final trained classification SVM model
%         .BestParams     : Structure with tuned parameters
%         .TrainAccuracy  : Training accuracy
%         .TestAccuracy   : Testing accuracy
%         .TestTrue       : True test labels
%         .TestPredictions: Predicted test labels
%         .gbestCost      : Final cost (CV loss) of best particle

%% Split Data into Training and Testing Sets
n = size(X,1);
% First split: training vs. remaining data
cv1 = cvpartition(n, 'HoldOut', 1 - TrainRatio);
XTrain = X(training(cv1),:);
yTrain = y(training(cv1));
XRemain = X(test(cv1),:);
yRemain = y(test(cv1));

% Second split: from remaining data, allocate TestRatio/(1-TrainRatio) for testing
nRemain = size(XRemain,1);
cv2 = cvpartition(nRemain, 'HoldOut', TestRatio/(1-TrainRatio));
XTest = XRemain(test(cv2),:);
yTest = yRemain(test(cv2));

%% PSO Setup for Classification SVM
% We tune 2 parameters:
%   1. BoxConstraint (search range: [0.1, 100])
%   2. KernelScale   (search range: [0.1, 10])
dim = 2;
lb = [0.1, 0.1];
ub = [100, 10];

% Initialize particles randomly within bounds
pos = repmat(lb, SwarmSize, 1) + rand(SwarmSize, dim) .* repmat(ub-lb, SwarmSize, 1);
vel = zeros(SwarmSize, dim);

% Evaluate initial cost for each particle using 5-fold cross-validation
pbest = pos;
pbestCost = inf(SwarmSize,1);
for i = 1:SwarmSize
    pbestCost(i) = costFunctionClassification(pos(i,:), XTrain, yTrain);
end

% Determine the global best
[gbestCost, idx] = min(pbestCost);
gbest = pos(idx,:);

% PSO parameters
w = 0.9;       % inertia weight (will be linearly decreased)
w_min = 0.4;   % minimum inertia weight
c1 = 2;        % cognitive coefficient
c2 = 2;        % social coefficient

%% PSO Main Loop
for iter = 1:MaxIterations
    % Linearly decrease inertia weight
    w = w - ((0.9 - w_min) / MaxIterations);
    for i = 1:SwarmSize
        r1 = rand(1, dim);
        r2 = rand(1, dim);
        % Update velocity and position
        vel(i,:) = w * vel(i,:) + c1 * r1 .* (pbest(i,:) - pos(i,:)) + c2 * r2 .* (gbest - pos(i,:));
        pos(i,:) = pos(i,:) + vel(i,:);
        % Enforce bounds
        pos(i,:) = max(pos(i,:), lb);
        pos(i,:) = min(pos(i,:), ub);
        
        % Evaluate cost at new position
        cost = costFunctionClassification(pos(i,:), XTrain, yTrain);
        if cost < pbestCost(i)
            pbest(i,:) = pos(i,:);
            pbestCost(i) = cost;
        end
        if cost < gbestCost
            gbest = pos(i,:);
            gbestCost = cost;
        end
    end
    fprintf('Iteration %d: Best CV Loss = %f\n', iter, gbestCost);
end

%% Train Final SVM Classification Model with Tuned Parameters
best_BoxConstraint = gbest(1);
best_KernelScale   = gbest(2);

% Note: 'Standardize' is set to true for robust performance.
svmModel = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', ...
    'BoxConstraint', best_BoxConstraint, 'KernelScale', best_KernelScale, ...
    'Standardize', true);

% Compute predictions and performance measures
yPredTrain = predict(svmModel, XTrain);
yPredTest  = predict(svmModel, XTest);
trainAcc   = sum(yPredTrain == yTrain) / numel(yTrain);
testAcc    = sum(yPredTest == yTest) / numel(yTest);

%% Return the Solution Structure
ClassificationSol.Model = svmModel;
ClassificationSol.BestParams = struct('BoxConstraint', best_BoxConstraint, ...
                                        'KernelScale', best_KernelScale);
ClassificationSol.TrainAccuracy = trainAcc;
ClassificationSol.TestAccuracy  = testAcc;
ClassificationSol.TestTrue      = yTest;
ClassificationSol.TestPredictions = yPredTest;
ClassificationSol.gbestCost     = gbestCost;

%% Nested Cost Function for Classification
    function err = costFunctionClassification(params, X_cv, y_cv)
        % Unpack parameters
        BoxConstraint = params(1);
        KernelScale   = params(2);
        % Use 5-fold cross-validation for classification SVM
        try
            cvMdl = fitcsvm(X_cv, y_cv, 'KernelFunction', 'rbf', ...
                'BoxConstraint', BoxConstraint, 'KernelScale', KernelScale, ...
                'Standardize', true, 'KFold', 5);
            err = kfoldLoss(cvMdl);
        catch ME
            % In case of error, assign a high cost.
            warning('Error in CV evaluation: %s', ME.message);
            err = inf;
        end
    end

end
