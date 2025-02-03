function [BestSol] = FA(CostFunction, nVar, maxIt, nPop, gamma, beta0, alpha, alpha_damp, m, lb, ub)
    % Parameters (now taken from input arguments)
    % alpha: Randomness strength
    % beta0: Attraction coefficient base value
    % gamma: Absorption coefficient
    % alpha_damp: Damping factor for alpha
    % m: Additional parameter (if used)

    % Ensure randomness
    rng shuffle

    % Initialize fireflies
    fireflies = repmat(lb, nPop, 1) + rand(nPop, nVar) .* (repmat(ub, nPop, 1) - repmat(lb, nPop, 1));
    
    % Evaluate initial population
    fitness = arrayfun(@(i) CostFunction(fireflies(i, :)), 1:nPop);
    
    % Find the best firefly
    [BestCost, idx] = min(fitness);
    BestSol.Position = fireflies(idx, :);
    BestSol.Cost = BestCost;
    
    % Main loop
    for it = 1:maxIt
        for i = 1:nPop
            for j = 1:nPop
                if fitness(j) < fitness(i)
                    % Calculate distance between fireflies i and j
                    r = norm(fireflies(i, :) - fireflies(j, :));
                    % Update position of firefly i
                    beta = beta0 * exp(-gamma * r^2);
                    fireflies(i, :) = fireflies(i, :) + beta * (fireflies(j, :) - fireflies(i, :)) + alpha * (rand(1, nVar) - 0.5);
                    % Apply bounds
                    fireflies(i, :) = max(fireflies(i, :), lb);
                    fireflies(i, :) = min(fireflies(i, :), ub);
                    % Evaluate new solution
                    fitness(i) = CostFunction(fireflies(i, :));
                end
            end
        end
        
        % Dampen the randomness parameter (alpha)
        alpha = alpha * alpha_damp;
        
        % Update the best solution
        [CurrentBestCost, idx] = min(fitness);
        if CurrentBestCost < BestSol.Cost
            BestSol.Position = fireflies(idx, :);
            BestSol.Cost = CurrentBestCost;
        end
        
        % Display iteration information
        disp(['Iteration ', num2str(it), ': Best Cost = ', num2str(BestSol.Cost), ' | Best Position = ', num2str(BestSol.Position)]);
    end
end