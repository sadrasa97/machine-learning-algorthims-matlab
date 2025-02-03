function sol = ClassifierSVM(NaturalFrequancy, DamageLocation, hiddenLayerSize, TrainRatio, TestRatio, isSymmetry, Symmetry)

%% Data Preparation
X = NaturalFrequancy;
Y = DamageLocation;

if size(X,1) < size(X,2)
    X = X';
end
if size(Y,1) < size(Y,2)
    Y = Y';
end

N = size(X,1);
idx = randperm(N);
X = X(idx,:);
Y = Y(idx);

nTrain = floor(TrainRatio * N);
Xtrain = X(1:nTrain,:);
Ytrain = Y(1:nTrain);
Xtest = X(nTrain+1:end,:);
Ytest = Y(nTrain+1:end);

%% SVM Classification Training Parameters
C = 1;       % Regularization parameter
lr = 1e-3;   % Learning rate
numEpochs = 100; % Number of training epochs

numFeatures = size(Xtrain,2);
w = zeros(numFeatures,1);
b = 0;

%% Training Loop (Gradient Descent using Hinge Loss)
for epoch = 1:numEpochs
    grad_w = zeros(numFeatures,1);
    grad_b = 0;
    
    % Loop over training samples (batch update)
    for i = 1:nTrain
        xi = Xtrain(i,:)';   % Column vector
        % Compute margin:
        margin = Ytrain(i) * (w' * xi + b);
        
        % Compute subgradient of the hinge loss:
        if margin < 1
            grad_loss = -Ytrain(i);
        else
            grad_loss = 0;
        end
        
        % Accumulate gradients (loss scaled by C)
        grad_w = grad_w + C * grad_loss * xi;
        grad_b = grad_b + C * grad_loss;
    end
    
    % Add regularization gradient
    grad_w = grad_w + w;
    
    % Update the parameters:
    w = w - lr * grad_w;
    b = b - lr * grad_b;
end

%% Testing: Compute predictions on the test set
scores = Xtest * w + b;
TestPredictions = ones(size(scores));
TestPredictions(scores < 0) = -1;

%% Package the results into the output structure
sol.model.w = w;
sol.model.b = b;
sol.TestTrue = Ytest;
sol.TestPredictions = TestPredictions;
end
