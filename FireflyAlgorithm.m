function bestParams = FireflyAlgorithm(fitnessFunc, nVars, lb, ub, MaxIterations, SwarmSize, initialGuess)
    % Parameters for Firefly Algorithm
    alpha = 0.2;  % Randomness factor
    beta0 = 1;    % Attractiveness at distance 0
    gamma = 1;    % Absorption coefficient

    % Initialize fireflies
    fireflies = repmat(initialGuess, SwarmSize, 1) + alpha * (rand(SwarmSize, nVars) - 0.5);
    fireflies = max(min(fireflies, ub), lb);  % Ensure bounds
    intensity = arrayfun(@(i) fitnessFunc(fireflies(i, :)), 1:SwarmSize);

    % FA iterations
    for iter = 1:MaxIterations
        for i = 1:SwarmSize
            for j = 1:SwarmSize
                if intensity(j) < intensity(i)
                    % Calculate distance between fireflies i and j
                    r = norm(fireflies(i, :) - fireflies(j, :));
                    
                    % Update attractiveness
                    beta = beta0 * exp(-gamma * r^2);
                    
                    % Move firefly i towards firefly j
                    fireflies(i, :) = fireflies(i, :) + beta * (fireflies(j, :) - fireflies(i, :)) + alpha * (rand(1, nVars) - 0.5);
                    
                    % Ensure bounds
                    fireflies(i, :) = max(min(fireflies(i, :), ub), lb);
                    
                    % Update intensity
                    intensity(i) = fitnessFunc(fireflies(i, :));
                end
            end
        end
    end

    % Find the best firefly
    [~, bestIdx] = min(intensity);
    bestParams = fireflies(bestIdx, :);
end
