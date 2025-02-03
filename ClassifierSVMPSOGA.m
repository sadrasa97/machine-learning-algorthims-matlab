
function ClassificationSol = ClassifierSVMPSOGA(NaturalFrequancy, DamageLocation, TrainRatio, TestRatio, MaxIterations, SwarmSize, isSymmetry, Symmetry)
    % Handle symmetry if applicable
    if isSymmetry
        DamageLocation = mapSymmetry(DamageLocation, Symmetry);
    end
    
    % Split data into training and testing sets
    cv = cvpartition(size(NaturalFrequancy,1), 'HoldOut', TestRatio);
    X_train = NaturalFrequancy(cv.training,:);
    y_train = DamageLocation(cv.training);
    X_test = NaturalFrequancy(cv.test,:);
    y_test = DamageLocation(cv.test);
    
    % Normalize features
    [X_train, mu, sigma] = zscore(X_train);
    X_test = (X_test - mu) ./ sigma;
    
    % Objective function for hyperparameter optimization
    objFunc = @(params) svmClassificationLoss(params, X_train, y_train);
    
    % PSO parameters
    nVars = 2; % [BoxConstraint, KernelScale]
    lb = [1e-3, 1e-3];
    ub = [1e3, 1e3];
    
    % Run PSO
    options_pso = optimoptions('particleswarm', 'SwarmSize', SwarmSize, ...
        'MaxIterations', MaxIterations, 'Display', 'iter');
    [pso_params, pso_loss] = particleswarm(objFunc, nVars, lb, ub, options_pso);
    
    % Run GA starting from PSO solution
    initialPopulation = repmat(pso_params, SwarmSize, 1) .* (1 + 0.1*randn(SwarmSize, nVars));
    options_ga = optimoptions('ga', 'InitialPopulationMatrix', initialPopulation, ...
        'MaxGenerations', MaxIterations, 'PopulationSize', SwarmSize, 'Display', 'iter');
    [ga_params, ga_loss] = ga(objFunc, nVars, [], [], [], [], lb, ub, [], options_ga);
    
    % Determine best parameters
    final_params = ga_params;
    
    % Train final SVM model
    svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', final_params(1), 'KernelScale', final_params(2));
    
    % Predict and evaluate
    y_pred = predict(svm_model, X_test);
    accuracy = sum(y_pred == y_test) / numel(y_test);
    confMat = confusionmat(y_test, y_pred);
    
    % Return results
    ClassificationSol.model = svm_model;
    ClassificationSol.accuracy = accuracy;
    ClassificationSol.confusionMatrix = confMat;
    ClassificationSol.params = final_params;
    ClassificationSol.output = y_pred;
end

function loss = svmClassificationLoss(params, X, y)
    C = params(1);
    sigma = params(2);

    t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', C, 'KernelScale', sigma, 'Standardize', false);
    mdl = fitcecoc(X, y, 'Learners', t, 'KFold', 5);
    loss = kfoldLoss(mdl); 
end


function dl = mapSymmetry(dl, Symmetry)
    for i = 1:size(Symmetry,1)
        for j = 2:size(Symmetry,2)
            dl(dl == Symmetry(i,j)) = Symmetry(i,1);
        end
    end
end
