function RegressionSol = regressionANN(X, Y, hiddenLayerSize, TrainRatio, TestRatio, varargin)
% regressionANN - Train a feed-forward ANN for regression from scratch.
%
% Syntax:
%   RegressionSol = regressionANN(X, Y, hiddenLayerSize, TrainRatio, TestRatio, Name, Value, ...)
%
% Inputs:
%   X              - n-by-d matrix of predictors.
%   Y              - n-by-1 vector of continuous responses.
%   hiddenLayerSize- Vector specifying the number of neurons in each hidden layer.
%   TrainRatio     - Fraction (0–1) of samples used for training.
%   TestRatio      - Fraction (0–1) of samples used for testing (the remainder is used for validation).
%
% Optional Name-Value pairs:
%   'MaxIter'      - Maximum number of iterations (default: 1000).
%   'LearningRate' - Learning rate for gradient descent (default: 1e-3).
%
% Outputs:
%   RegressionSol  - A structure with fields:
%         Model            - A struct containing the learned weights, biases, and a prediction function.
%         TrainPredictions - Predictions on the training set.
%         TestPredictions  - Predictions on the test set.
%         ValPredictions   - Predictions on the validation set.
%         TrainRMSE, TestRMSE, ValRMSE - RMSE on each set.
%         TrainR2, TestR2, ValR2       - Coefficient of determination on each set.
%
% Example:
%   sol = regressionANN(X, Y, [10 10], 0.7, 0.15, 'MaxIter', 2000, 'LearningRate', 1e-4);

    %% Parse optional parameters
    p = inputParser;
    addParameter(p, 'MaxIter', 1000, @isnumeric);
    addParameter(p, 'LearningRate', 1e-3, @isnumeric);
    parse(p, varargin{:});
    maxIter = p.Results.MaxIter;
    lr = p.Results.LearningRate;
    
    %% Data splitting
    n = size(X,1);
    idx = randperm(n);
    nTrain = round(TrainRatio * n);
    nTest  = round(TestRatio  * n);
    trainIdx = idx(1:nTrain);
    testIdx  = idx(nTrain+1:nTrain+nTest);
    valIdx   = idx(nTrain+nTest+1:end);
    
    Xtrain = X(trainIdx, :);
    Ytrain = Y(trainIdx);
    Xtest  = X(testIdx, :);
    Ytest  = Y(testIdx);
    Xval   = X(valIdx, :);
    Yval   = Y(valIdx);
    
    %% Network architecture
    inputSize = size(X,2);
    outputSize = 1;  % regression output is a scalar
    layers = [inputSize, hiddenLayerSize, outputSize];
    L = numel(layers)-1;  % number of weight layers
    
    %% Initialize weights and biases (using small random values)
    rng(0);  % for reproducibility
    W = cell(L,1);
    b = cell(L,1);
    for i = 1:L
        W{i} = randn(layers(i), layers(i+1)) * 0.1;
        b{i} = zeros(1, layers(i+1));
    end
    
    %% Training via full–batch gradient descent
    m = size(Xtrain,1);
    for iter = 1:maxIter
        % Forward pass
        a = cell(L+1,1);
        z = cell(L,1);
        a{1} = Xtrain;  % input layer
        
        for i = 1:L-1
            z{i} = a{i} * W{i} + repmat(b{i}, m, 1);
            a{i+1} = tanh(z{i});  % tanh activation in hidden layers
        end
        % Final layer: linear activation
        z{L} = a{L} * W{L} + repmat(b{L}, m, 1);
        a{L+1} = z{L};
        
        % Compute mean squared error loss
        diff = a{L+1} - Ytrain;
        loss = 0.5 * mean(diff.^2);
        
        % Backward pass (compute gradients)
        % Derivative of loss wrt final output (linear layer)
        delta = diff / m;  % m-by-1
        
        dW = cell(L,1);
        db = cell(L,1);
        % For the final layer
        dW{L} = a{L}' * delta;
        db{L} = sum(delta, 1);
        
        % Backpropagate through hidden layers
        for i = L-1:-1:1
            % derivative of tanh is: 1 - tanh(z)^2 = 1 - a{i+1}.^2
            delta = (delta * W{i+1}') .* (1 - a{i+1}.^2);
            dW{i} = a{i}' * delta;
            db{i} = sum(delta, 1);
        end
        
        % Update weights and biases
        for i = 1:L
            W{i} = W{i} - lr * dW{i};
            b{i} = b{i} - lr * db{i};
        end
        
        % Optionally display loss every 100 iterations
        if mod(iter,100)==0 || iter==1
            fprintf('Iteration %d: Training Loss = %.6f\n', iter, loss);
        end
    end
    
    %% Build a model structure with a prediction function
    Model.W = W;
    Model.b = b;
    Model.predict = @(Xinput) nnForward(Xinput, Model);
    
    %% Obtain predictions
    Y_train_pred = Model.predict(Xtrain);
    Y_test_pred  = Model.predict(Xtest);
    Y_val_pred   = Model.predict(Xval);
    
    %% Compute metrics
    [trainRMSE, trainR2] = calculateMetrics(Ytrain, Y_train_pred);
    [testRMSE, testR2]   = calculateMetrics(Ytest, Y_test_pred);
    [valRMSE, valR2]     = calculateMetrics(Yval, Y_val_pred);
    
    %% Package results
    RegressionSol = struct('Model', Model, ...
                           'TrainPredictions', Y_train_pred, ...
                           'TestPredictions', Y_test_pred, ...
                           'ValPredictions', Y_val_pred, ...
                           'TrainRMSE', trainRMSE, ...
                           'TestRMSE', testRMSE, ...
                           'ValRMSE', valRMSE, ...
                           'TrainR2', trainR2, ...
                           'TestR2', testR2, ...
                           'ValR2', valR2);
                       
    fprintf('Final Regression Metrics:\n');
    fprintf('Training RMSE: %.6f, R²: %.6f\n', trainRMSE, trainR2);
    fprintf('Testing RMSE:  %.6f, R²: %.6f\n', testRMSE, testR2);
    fprintf('Validation RMSE: %.6f, R²: %.6f\n', valRMSE, valR2);
end

%% Helper function: forward propagation using learned weights
function Ypred = nnForward(Xinput, Model)
    W = Model.W;
    b = Model.b;
    L = numel(W);
    m = size(Xinput,1);
    a = Xinput;
    for i = 1:L-1
        z = a * W{i} + repmat(b{i}, m, 1);
        a = tanh(z);
    end
    % Final layer (linear)
    z = a * W{L} + repmat(b{L}, m, 1);
    Ypred = z;
end

%% Helper function: calculate RMSE and R²
function [rmse, r2] = calculateMetrics(Y_true, Y_pred)
    rmse = sqrt(mean((Y_true - Y_pred).^2));
    ss_tot = sum((Y_true - mean(Y_true)).^2);
    ss_res = sum((Y_true - Y_pred).^2);
    r2 = 1 - (ss_res/ss_tot);
end
