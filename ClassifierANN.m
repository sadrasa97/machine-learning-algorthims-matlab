function ClassificationSol = ClassifierANN(X, Y, hiddenLayerSize, TrainRatio, TestRatio, isSymmetry, Symmetry, varargin)
% ClassifierANN - Train a feed-forward ANN for classification using ReLU hidden layers.
%
% Syntax:
%   ClassificationSol = ClassifierANN(X, Y, hiddenLayerSize, TrainRatio, TestRatio, isSymmetry, Symmetry, Name, Value, ...)
%
% Inputs:
%   X              - n-by-d matrix of features.
%   Y              - n-by-1 vector of labels.
%   hiddenLayerSize- Vector specifying the number of neurons in each hidden layer.
%   TrainRatio     - Fraction (0–1) of samples used for training.
%   TestRatio      - Fraction (0–1) of samples used for testing (the remainder is for validation).
%   isSymmetry     - (Optional) Boolean flag indicating if symmetry mapping is needed.
%   Symmetry       - (Optional) Matrix defining symmetry mapping of labels.
%
% Optional Name-Value pairs:
%   'MaxIter'      - Maximum number of iterations (default: 1000).
%   'LearningRate' - Learning rate for gradient descent (default: 1e-3).
%
% Outputs:
%   ClassificationSol - A structure with fields:
%         Model            - A struct containing the learned weights, biases, and a prediction function.
%         TrainPredictions - Predicted labels for the training set.
%         TestPredictions  - Predicted labels for the test set.
%         ValPredictions   - Predicted labels for the validation set.
%         TrainAccuracy, TestAccuracy, ValAccuracy - Accuracy on each set.
%         TestTrue         - The true labels for the test set.
%
% Example:
%   sol = ClassifierANN(X, Y, [10 10], 0.7, 0.15, false, [], 'MaxIter', 1500, 'LearningRate', 1e-3);

    %% Parse optional parameters
    p = inputParser;
    addParameter(p, 'MaxIter', 1000, @isnumeric);
    addParameter(p, 'LearningRate', 1e-3, @isnumeric);
    parse(p, varargin{:});
    maxIter = p.Results.MaxIter;
    lr = p.Results.LearningRate;
    
    %% Optional symmetry mapping
    if nargin >= 6 && isSymmetry && ~isempty(Symmetry)
        for i = 1:size(Symmetry,1)
            for j = 2:size(Symmetry,2)
                Y(Y==Symmetry(i,j)) = Symmetry(i,1);
            end
        end
    end
    
    %% Convert labels to consecutive integers and one-hot encode
    classLabels = unique(Y);
    numClasses = numel(classLabels);
    label2idx = containers.Map(num2cell(classLabels), num2cell(1:numClasses));
    Yidx = arrayfun(@(v) label2idx(v), Y);
    
    nSamples = length(Y);
    Y_onehot = zeros(nSamples, numClasses);
    for i = 1:nSamples
        Y_onehot(i, Yidx(i)) = 1;
    end
    
    %% Data splitting
    nTotal = size(X,1);
    idx = randperm(nTotal);
    nTrain = round(TrainRatio * nTotal);
    nTest  = round(TestRatio  * nTotal);
    trainIdx = idx(1:nTrain);
    testIdx  = idx(nTrain+1:nTrain+nTest);
    valIdx   = idx(nTrain+nTest+1:end);
    
    Xtrain = X(trainIdx, :);
    Ytrain_onehot = Y_onehot(trainIdx, :);
    Ytrain_idx = Yidx(trainIdx);
    
    Xtest = X(testIdx, :);
    Ytest_onehot = Y_onehot(testIdx, :);
    Ytest_idx = Yidx(testIdx);
    
    Xval = X(valIdx, :);
    Yval_onehot = Y_onehot(valIdx, :);
    Yval_idx = Yidx(valIdx);
    
    %% Network architecture
    inputSize = size(X,2);
    outputSize = numClasses;
    layers = [inputSize, hiddenLayerSize, outputSize];
    L = numel(layers)-1;  % number of weight layers
    
    %% Initialize weights and biases (He initialization for ReLU)
    rng(0);  % reproducibility
    W = cell(L,1);
    b = cell(L,1);
    for i = 1:L
        % He initialization: scale = sqrt(2/number_of_inputs)
        scale = sqrt(2/layers(i));
        W{i} = randn(layers(i), layers(i+1)) * scale;
        b{i} = zeros(1, layers(i+1));
    end
    
    %% Training via full-batch gradient descent
    m = size(Xtrain,1);
    for iter = 1:maxIter
        % Forward pass
        a = cell(L+1,1);
        z = cell(L,1);
        a{1} = Xtrain;
        % Hidden layers: use ReLU activation
        for i = 1:L-1
            z{i} = a{i} * W{i} + repmat(b{i}, m, 1);
            a{i+1} = max(0, z{i});  % ReLU activation
        end
        % Output layer: softmax activation
        z{L} = a{L} * W{L} + repmat(b{L}, m, 1);
        a{L+1} = softmax(z{L});
        
        % Compute cross-entropy loss (with a small constant for stability)
        loss = -mean(sum(Ytrain_onehot .* log(a{L+1} + 1e-10), 2));
        
        % Backward pass
        % For softmax + cross-entropy, the gradient is:
        delta = (a{L+1} - Ytrain_onehot);  % m-by-outputSize
        
        dW = cell(L,1);
        db = cell(L,1);
        dW{L} = a{L}' * delta / m;
        db{L} = sum(delta, 1) / m;
        
        % Backpropagate through hidden layers
        for i = L-1:-1:1
            % Derivative for ReLU: 1 if z > 0, else 0
            dReLU = double(z{i} > 0);
            delta = (delta * W{i+1}') .* dReLU;
            dW{i} = a{i}' * delta / m;
            db{i} = sum(delta, 1) / m;
        end
        
        % Update weights and biases
        for i = 1:L
            W{i} = W{i} - lr * dW{i};
            b{i} = b{i} - lr * db{i};
        end
        
        % Every 100 iterations, report loss and training accuracy
        if mod(iter,100)==0 || iter==1
            % Compute training predictions and accuracy
            trainScores = annPredictClassifier(Xtrain, W, b);
            [~, trainPredIdx] = max(trainScores, [], 2);
            trainAcc = mean(trainPredIdx == Ytrain_idx);
            fprintf('Iteration %d: Training Loss = %.6f, Training Accuracy = %.4f\n', iter, loss, trainAcc);
        end
    end
    
    %% Build model structure with prediction function
    Model.W = W;
    Model.b = b;
    Model.predict = @(Xinput) annPredictClassifier(Xinput, W, b);
    
    %% Predictions on each set
    trainScores = Model.predict(Xtrain);
    [~, trainPredIdx] = max(trainScores, [], 2);
    testScores = Model.predict(Xtest);
    [~, testPredIdx] = max(testScores, [], 2);
    valScores = Model.predict(Xval);
    [~, valPredIdx] = max(valScores, [], 2);
    
    % Map indices back to original labels
    idx2label = containers.Map(num2cell(1:numClasses), num2cell(classLabels));
    trainPredLabels = arrayfun(@(i) idx2label(i), trainPredIdx);
    testPredLabels  = arrayfun(@(i) idx2label(i), testPredIdx);
    valPredLabels   = arrayfun(@(i) idx2label(i), valPredIdx);
    
    %% Compute accuracies
    trainAcc = mean(trainPredIdx == Ytrain_idx);
    testAcc  = mean(testPredIdx == Ytest_idx);
    valAcc   = mean(valPredIdx == Yval_idx);
    
    %% Package results
    ClassificationSol = struct('Model', Model, ...
                               'TrainPredictions', trainPredLabels, ...
                               'TestPredictions', testPredLabels, ...
                               'ValPredictions', valPredLabels, ...
                               'TrainAccuracy', trainAcc, ...
                               'TestAccuracy', testAcc, ...
                               'ValAccuracy', valAcc, ...
                               'TestTrue', Y(testIdx));
    

end

%% Softmax function (applied row-wise)
function s = softmax(z)
    % For numerical stability subtract the row maximum
    z = bsxfun(@minus, z, max(z,[],2));
    expz = exp(z);
    s = bsxfun(@rdivide, expz, sum(expz,2));
end

%% Prediction function for the classifier network
function scores = annPredictClassifier(Xinput, W, b)
    L = numel(W);
    m = size(Xinput,1);
    a = Xinput;
    for i = 1:L-1
        z = a * W{i} + repmat(b{i}, m, 1);
        a = max(0, z);  % ReLU activation
    end
    z = a * W{L} + repmat(b{L}, m, 1);
    scores = softmax(z);
end
