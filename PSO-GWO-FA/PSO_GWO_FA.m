function BestSol = PSO_GWO_FA(CostFunction, nVar, lb, ub, nPop, maxIt, ...
                              w, c1, c2, wDamp, a, a_damp, gamma, beta0, alpha, alpha_damp)

    particles = repmat(struct('Position', [], 'Velocity', [], 'Cost', [], 'pBest', [], 'pBestCost', []), nPop, 1);
    for i = 1:nPop
        particles(i).Position = lb + rand(1, nVar) .* (ub - lb);
        particles(i).Velocity = zeros(1, nVar);
        particles(i).Cost = CostFunction(particles(i).Position);
        particles(i).pBest = particles(i).Position;
        particles(i).pBestCost = particles(i).Cost;
    end

    % Initialize GWO Alpha, Beta, Delta
    [~, sortedIdx] = sort([particles.Cost]);
    alpha_wolf = particles(sortedIdx(1));
    beta_wolf = particles(sortedIdx(2));
    delta_wolf = particles(sortedIdx(3));

    % Initialize PSO Global Best
    [~, idx] = min([particles.Cost]);
    gBest = particles(idx);

    % Main Loop
    for it = 1:maxIt

        for i = 1:nPop
            % Update Velocity
            r1 = rand(1, nVar);
            r2 = rand(1, nVar);
            particles(i).Velocity = w * particles(i).Velocity ...
                + c1 * r1 .* (particles(i).pBest - particles(i).Position) ...
                + c2 * r2 .* (gBest.Position - particles(i).Position);
            
            % Update Position
            particles(i).Position = particles(i).Position + particles(i).Velocity;
            particles(i).Position = max(min(particles(i).Position, ub), lb);
            particles(i).Cost = CostFunction(particles(i).Position);
            
            % Update Personal Best
            if particles(i).Cost < particles(i).pBestCost
                particles(i).pBest = particles(i).Position;
                particles(i).pBestCost = particles(i).Cost;
            end
        end


        a_val = 2 * a * (1 - (it/maxIt)); % Linearly decrease exploration
        for i = 1:nPop
            % Update positions using Alpha, Beta, Delta
            X1 = alpha_wolf.Position - a_val * rand(1, nVar) .* abs(alpha_wolf.Position - particles(i).Position);
            X2 = beta_wolf.Position - a_val * rand(1, nVar) .* abs(beta_wolf.Position - particles(i).Position);
            X3 = delta_wolf.Position - a_val * rand(1, nVar) .* abs(delta_wolf.Position - particles(i).Position);
            newPos = (X1 + X2 + X3) / 3;
            
            % Apply GWO Update
            particles(i).Position = max(min(newPos, ub), lb);
            particles(i).Cost = CostFunction(particles(i).Position);
        end

        for i = 1:nPop
            for j = 1:nPop
                if particles(j).Cost < particles(i).Cost
                    % Calculate distance and attractiveness
                    r = norm(particles(i).Position - particles(j).Position);
                    beta = beta0 * exp(-gamma * r^2);
                    
                    % Update position using FA
                    particles(i).Position = particles(i).Position ...
                        + beta * (particles(j).Position - particles(i).Position) ...
                        + alpha * (rand(1, nVar) - 0.5);
                    particles(i).Position = max(min(particles(i).Position, ub), lb);
                    particles(i).Cost = CostFunction(particles(i).Position);
                end
            end
        end


        % Update GWO Alpha, Beta, Delta
        [~, sortedIdx] = sort([particles.Cost]);
        alpha_wolf = particles(sortedIdx(1));
        beta_wolf = particles(sortedIdx(2));
        delta_wolf = particles(sortedIdx(3));
        
        % Update PSO Global Best
        [~, idx] = min([particles.pBestCost]);
        gBest = particles(idx);


        w = w * wDamp;          % Damp PSO inertia
        a = a * a_damp;         % Damp GWO exploration
        alpha = alpha * alpha_damp; % Damp FA randomness
    end

    BestSol = gBest;
end