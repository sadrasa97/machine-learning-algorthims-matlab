function RegressionSol = RegressionSVMPSO(...
    X, y, TrainRatio, TestRatio, ...
    MaxPSOIter, SwarmSize, maxIterSVR, learningRate ...
)
% RegressionSVMPSO tunes a custom linear SVR using PSO for regression.
%
% INPUTS:
%   X            - (n x d) feature matrix.
%   y            - (n x 1) target vector.
%   TrainRatio   - Fraction of data used for training (e.g., 0.7).
%   TestRatio    - Fraction of data used for testing (e.g., 0.15).
%   MaxPSOIter   - Maximum number of PSO iterations.
%   SwarmSize    - Number of particles in the PSO swarm.
%   maxIterSVR   - Number of iterations for SVR training.
%   learningRate - Learning rate for SVR training.
%
% OUTPUT:
%   RegressionSol - Structure containing:
%                   Model      : Final trained SVR model.
%                   BestParams : Tuned parameters C and epsilon.
%                   TrainMSE   : Training mean squared error.
%                   TestMSE    : Testing mean squared error.
%                   gbestCost  : Best (lowest) CV loss achieved during PSO.

    % Ensure y is a column vector
    y = y(:);
    
    %% Split Data into Training and Testing Sets
    n = size(X,1);
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
    
    %% PSO Setup for Tuning SVR Hyperparameters
    dim = 2;  % Two parameters: C and epsilon
    lb = [0.1, 0.001];
    ub = [100, 1];
    
    % Initialize particles within bounds
    pos = lb + rand(SwarmSize, dim) .* (ub - lb);
    vel = zeros(SwarmSize, dim);
    
    % Evaluate initial cost for each particle using 5-fold CV
    pbest = pos;
    pbestCost = inf(SwarmSize, 1);
    for i = 1:SwarmSize
        pbestCost(i) = costFunctionSVR(pos(i,:), XTrain, yTrain, maxIterSVR, learningRate);
    end
    
    % Determine global best
    [gbestCost, idx] = min(pbestCost);
    gbest = pos(idx,:);
    
    % PSO parameters
    w = 0.9;       % Initial inertia weight
    w_min = 0.4;   % Minimum inertia weight
    c1 = 2;        % Cognitive coefficient
    c2 = 2;        % Social coefficient
    
    %% PSO Main Loop
    for iter = 1:MaxPSOIter
        % Linearly decrease inertia weight
        w = w - ((0.9 - w_min) / MaxPSOIter);
        
        for i = 1:SwarmSize
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            
            % Velocity update
            vel(i,:) = w * vel(i,:) + c1 * r1 .* (pbest(i,:) - pos(i,:)) + c2 * r2 .* (gbest - pos(i,:));
            
            % Position update with bounds enforcement
            pos(i,:) = max(min(pos(i,:) + vel(i,:), ub), lb);
            
            % Evaluate cost at new position
            cost = costFunctionSVR(pos(i,:), XTrain, yTrain, maxIterSVR, learningRate);
            if cost < pbestCost(i)
                pbest(i,:) = pos(i,:);
                pbestCost(i) = cost;
            end
            if cost < gbestCost
                gbest = pos(i,:);
                gbestCost = cost;
            end
        end
        fprintf('PSO Iteration %d: Best CV Loss = %f\n', iter, gbestCost);
    end

    %% Final Training with the Best Parameters on Full Training Set
    best_C = gbest(1);
    best_epsilon = gbest(2);
    
    finalModel = CustomSVRTrain(XTrain, yTrain, best_C, best_epsilon, learningRate, maxIterSVR);
    
    % Compute predictions and MSE on training and testing data.
    yPredTrain = CustomSVRPredict(finalModel, XTrain);
    yPredTest  = CustomSVRPredict(finalModel, XTest);
    trainMSE = mean((yTrain - yPredTrain).^2);
    testMSE  = mean((yTest - yPredTest).^2);
    
    %% Return the solution structure
    RegressionSol.Model = finalModel;
    RegressionSol.BestParams = struct('C', best_C, 'epsilon', best_epsilon);
    RegressionSol.TrainMSE = trainMSE;
    RegressionSol.TestMSE = testMSE;
    RegressionSol.gbestCost = gbestCost;
end

%% Custom SVR Training Function
function model = CustomSVRTrain(X, y, C, epsilon, lr, maxIter)
    % Custom SVR training using sub-gradient descent
    [n, d] = size(X);
    w = zeros(d, 1);
    b = 0;
    
    for iter = 1:maxIter
        y_pred = X * w + b;
        errors = y - y_pred;
        abs_errors = abs(errors);
        
        % Compute subgradients
        mask = abs_errors >= epsilon;
        dw = (1/n) * (X' * (sign(errors) .* mask)) - (1/(C*n)) * w;
        db = (1/n) * sum(sign(errors) .* mask);
        
        % Update parameters
        w = w + lr * dw;
        b = b + lr * db;
    end
    
    model.w = w;
    model.b = b;
    model.C = C;
    model.epsilon = epsilon;
end

%% Custom SVR Prediction Function
function y_pred = CustomSVRPredict(model, X)
    y_pred = X * model.w + model.b;
end

%% Cost Function for PSO Optimization (5-fold CV)
function mse = costFunctionSVR(params, X_cv, y_cv, maxIterLocal, lr)
    C_local = params(1);
    epsilon_local = params(2);
    
    K = 5;  % 5-fold cross-validation
    cvp = cvpartition(size(X_cv,1), 'KFold', K);
    mse_folds = zeros(K,1);
    
    for k = 1:K
        Xtr = X_cv(training(cvp, k), :);
        ytr = y_cv(training(cvp, k));
        Xval = X_cv(test(cvp, k), :);
        yval = y_cv(test(cvp, k));
        
        % Train SVR model
        modelLocal = CustomSVRTrain(Xtr, ytr, C_local, epsilon_local, lr, maxIterLocal);
        yPredLocal = CustomSVRPredict(modelLocal, Xval);
        mse_folds(k) = mean((yval - yPredLocal).^2);
    end
    mse = mean(mse_folds);
end
