function sol = RegressionSVM(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio)

%% Data Preparation
X = NaturalFrequancy;
Y = DamageRatio;
N = size(X,1);
idx = randperm(N);
X = X(idx,:);
Y = Y(idx);

nTrain = floor(TrainRatio * N);
Xtrain = X(1:nTrain,:);
Ytrain = Y(1:nTrain);
Xtest = X(nTrain+1:end,:);
Ytest = Y(nTrain+1:end);

%% SVM Regression Training Parameters
C = 1;           % Regularization parameter
epsilon = 0.1;   % Epsilon-insensitive zone
lr = 1e-3;       % Learning rate
numEpochs = 100; % Number of training epochs

numFeatures = size(Xtrain,2);
w = zeros(numFeatures,1);
b = 0;

%% Training Loop (Gradient Descent)
for epoch = 1:numEpochs
    grad_w = zeros(numFeatures,1);
    grad_b = 0;
    
    % Loop over training samples (batch update)
    for i = 1:nTrain
        xi = Xtrain(i,:)';   % Column vector
        % Compute prediction and error:
        y_pred = w' * xi + b;
        error = y_pred - Ytrain(i);
        
        % Compute subgradient of the epsilon-insensitive loss:
        if abs(error) > epsilon
            grad_loss = sign(error);
        else
            grad_loss = 0;
        end
        
        % Accumulate gradients (with SVM loss scaled by C)
        grad_w = grad_w + C * grad_loss * xi;
        grad_b = grad_b + C * grad_loss;
    end
    
    % Add regularization gradients (using an ?? penalty)
    grad_w = grad_w + w;
    
    % Update the parameters:
    w = w - lr * grad_w;
    b = b - lr * grad_b;
end

%% Testing: Compute predictions on the test set
TestPredictions = Xtest * w + b;

%% Package the results into the output structure
sol.model.w = w;
sol.model.b = b;
sol.TestTrue = Ytest;
sol.TestPredictions = TestPredictions;
end


