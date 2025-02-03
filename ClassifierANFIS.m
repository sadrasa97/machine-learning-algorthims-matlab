function sol = ClassifierANFIS(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, isSymmetry, Symmetry)
    %% Data Preparation: Split data into training and testing sets
    X = NaturalFrequancy;
    Y = DamageLocation;
    N = size(X,1);
    idx = randperm(N);
    X = X(idx,:);
    Y = Y(idx);
    nTrain = floor(TrainRatio * N);
    Xtrain = X(1:nTrain,:);
    Ytrain = Y(1:nTrain);
    Xtest = X(nTrain+1:end,:);
    Ytest = Y(nTrain+1:end);

    %% Normalize input features using z-score normalization
    [Xtrain, mu_X, sigma_X] = zscore(Xtrain);
    Xtest = (Xtest - mu_X) ./ sigma_X;

    %% Build the ANFIS Model Structure
    d = size(Xtrain,2);   % number of inputs
    M = 2;                % number of membership functions per input
    R = M^d;              % total number of rules

    % Initialize premise parameters
    centers = zeros(R,d);
    sigmas  = zeros(R,d);
    Ctemp = cell(1,d);
    for j = 1:d
        minVal = min(Xtrain(:,j));
        maxVal = max(Xtrain(:,j));
        centers_j = [minVal, maxVal];
        Ctemp{j} = centers_j;
        sigmas(:,j) = (maxVal - minVal) / 2;
    end
    [grid{1:d}] = ndgrid(Ctemp{:});
    for j = 1:d
        centers(:,j) = reshape(grid{j}, [], 1);
    end

    % Initialize consequent parameters: one linear function per rule
    p = randn(R, d+1) * 0.1;

    %% Training Parameters
    numEpochs = 100;
    lr_conseq = 1e-4;
    lr_premise = 1e-4;
    maxGradNorm = 1e2;

    %% Training Loop
    for epoch = 1:numEpochs
        totalError = 0;
        grad_p = zeros(size(p));
        grad_centers = zeros(size(centers));
        grad_sigmas  = zeros(size(sigmas));

        for i = 1:nTrain
            x = Xtrain(i,:)';
            y_true = Ytrain(i);

            % Compute firing strengths for each rule
            w = zeros(R,1);
            for r = 1:R
                prod_r = 1;
                for j = 1:d
                    sigma_val = max(sigmas(r,j), 1e-6);
                    prod_r = prod_r * exp( -((x(j) - centers(r,j))^2) / (2*sigma_val^2) );
                end
                w(r) = prod_r;
            end
            w_sum = sum(w);
            if w_sum == 0, w_sum = eps; end
            w_norm = w / w_sum;

            % Compute rule outputs (linear functions)
            f_rule = zeros(R,1);
            for r = 1:R
                f_rule(r) = p(r,1:d)*x + p(r,d+1);
            end

            % Overall continuous output
            f_out = sum(w_norm .* f_rule);
            error_i = f_out - y_true;
            totalError = totalError + 0.5 * error_i^2;

            % Gradient for consequent parameters
            for r = 1:R
                grad_p(r,1:d) = grad_p(r,1:d) + error_i * w_norm(r) * x';
                grad_p(r,d+1) = grad_p(r,d+1) + error_i * w_norm(r);
            end

            % Gradient for premise parameters
            for r = 1:R
                for j = 1:d
                    sigma_val = max(sigmas(r,j), 1e-6);
                    diff = x(j) - centers(r,j);
                    mu = exp( - (diff^2) / (2*sigma_val^2) );
                    dmu_dc = mu * ( diff / (sigma_val^2) );
                    dmu_dsigma = mu * ( diff^2 / (sigma_val^3) );
                    grad_centers(r,j) = grad_centers(r,j) + error_i * (f_rule(r) - f_out) * dmu_dc / w_sum;
                    grad_sigmas(r,j)  = grad_sigmas(r,j)  + error_i * (f_rule(r) - f_out) * dmu_dsigma / w_sum;
                end
            end
        end

        % Clip gradients for stability
        if norm(grad_p, 'fro') > maxGradNorm
            grad_p = grad_p * (maxGradNorm / norm(grad_p, 'fro'));
        end
        if norm(grad_centers, 'fro') > maxGradNorm
            grad_centers = grad_centers * (maxGradNorm / norm(grad_centers, 'fro'));
        end
        if norm(grad_sigmas, 'fro') > maxGradNorm
            grad_sigmas = grad_sigmas * (maxGradNorm / norm(grad_sigmas, 'fro'));
        end

        % Update parameters using averaged gradients
        p = p - lr_conseq * (grad_p / nTrain);
        centers = centers - lr_premise * (grad_centers / nTrain);
        sigmas = sigmas - lr_premise * (grad_sigmas / nTrain);

        % Constrain sigmas to remain positive
        sigmas = max(sigmas, 1e-6);

        if mod(epoch,10)==0
            fprintf('Epoch %d/%d, Training MSE = %.4f\n', epoch, numEpochs, totalError/nTrain);
        end
    end

    %% Store the Learned Model
    model.centers = centers;
    model.sigmas  = sigmas;
    model.p       = p;
    model.d       = d;
    model.R       = R;
    model.mu_X    = mu_X;
    model.sigma_X = sigma_X;

    %% Testing: Compute continuous outputs and then threshold
    numTest = size(Xtest,1);
    predictions_cont = zeros(numTest,1);
    for i = 1:numTest
        x = Xtest(i,:)';
        w = zeros(R,1);
        for r = 1:R
            prod_r = 1;
            for j = 1:d
                sigma_val = max(sigmas(r,j), 1e-6);
                prod_r = prod_r * exp( -((x(j)-centers(r,j))^2)/(2*sigma_val^2) );
            end
            w(r) = prod_r;
        end
        w_sum = sum(w);
        if w_sum == 0, w_sum = eps; end
        w_norm = w / w_sum;
        f_rule = zeros(R,1);
        for r = 1:R
            f_rule(r) = p(r,1:d)*x + p(r,d+1);
        end
        predictions_cont(i) = sum(w_norm .* f_rule);
    end

    % Threshold: assign +1 if output >= 0, else -1.
    predictions = ones(numTest,1);
    predictions(predictions_cont < 0) = -1;

    %% Prepare Output Structure
    sol.model = model;
    sol.TestTrue = Ytest;
    sol.TestPredictions = predictions;
end
