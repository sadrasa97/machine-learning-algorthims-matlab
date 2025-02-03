function Sol = PSO(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp)
    % Parameters (now passed from main script)
    % w: Initial inertia weight
    % c1: Personal learning coefficient
    % c2: Global learning coefficient
    % wDamp: Inertia weight damping ratio

    % Initialize population
    Particles = rand(nPop, nVar) .* (ub - lb) + lb;
    Velocities = zeros(nPop, nVar);
    
    % Initialize personal bests
    PersonalBest = Particles;  % Track each particle's best position
    PersonalBestCost = arrayfun(@(i) CostFunction(Particles(i, :)), 1:nPop);
    
    % Find initial global best
    [GlobalBestCost, bestIdx] = min(PersonalBestCost);
    GlobalBest = Particles(bestIdx, :);
    
    % Main PSO loop
    for it = 1:maxIt
        for i = 1:nPop
            % Update velocity with personal and global bests
            r1 = rand(1, nVar);
            r2 = rand(1, nVar);
            Velocities(i, :) = w * Velocities(i, :) ...
                + c1 * r1 .* (PersonalBest(i, :) - Particles(i, :)) ...
                + c2 * r2 .* (GlobalBest - Particles(i, :));
            
            % Update position and apply bounds
            Particles(i, :) = Particles(i, :) + Velocities(i, :);
            Particles(i, :) = max(min(Particles(i, :), ub), lb);
            
            % Evaluate new cost
            currentCost = CostFunction(Particles(i, :));
            
            % Update personal best if improved
            if currentCost < PersonalBestCost(i)
                PersonalBest(i, :) = Particles(i, :);
                PersonalBestCost(i) = currentCost;
                
                % Update global best if necessary
                if currentCost < GlobalBestCost
                    GlobalBest = Particles(i, :);
                    GlobalBestCost = currentCost;
                end
            end
        end
        
        % Apply inertia weight damping
        w = w * wDamp;
    end
    
    % Return best solution
    Sol.Position = GlobalBest;
    Sol.Cost = GlobalBestCost;
end