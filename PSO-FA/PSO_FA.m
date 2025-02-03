function BestSol = PSO_FA(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, gamma, beta0, alpha, alpha_damp)
    % Parameters:
    % - PSO Parameters: w, c1, c2, wDamp
    % - FA Parameters: gamma, beta0, alpha, alpha_damp

    % Initialize Particles/Fireflies
    particles = repmat(struct('Position', [], 'Velocity', [], 'Cost', [], 'pBest', [], 'pBestCost', []), nPop, 1);
    for i = 1:nPop
        particles(i).Position = lb + rand(1, nVar) .* (ub - lb);
        particles(i).Velocity = zeros(1, nVar);
        particles(i).Cost = CostFunction(particles(i).Position);
        particles(i).pBest = particles(i).Position;
        particles(i).pBestCost = particles(i).Cost;
    end

    % Initialize Global Best (PSO)
    [~, idx] = min([particles.Cost]);
    gBest = particles(idx);

    % Main Loop
    for it = 1:maxIt
        % PSO Velocity Update
        for i = 1:nPop
            r1 = rand(1, nVar);
            r2 = rand(1, nVar);
            particles(i).Velocity = w * particles(i).Velocity ...
                + c1 * r1 .* (particles(i).pBest - particles(i).Position) ...
                + c2 * r2 .* (gBest.Position - particles(i).Position);
            
            % Update Position with PSO
            particles(i).Position = particles(i).Position + particles(i).Velocity;
            particles(i).Position = max(min(particles(i).Position, ub), lb);
            particles(i).Cost = CostFunction(particles(i).Position);
            
            % Update Personal Best
            if particles(i).Cost < particles(i).pBestCost
                particles(i).pBest = particles(i).Position;
                particles(i).pBestCost = particles(i).Cost;
            end
        end
        
        % Firefly Attraction (FA)
        for i = 1:nPop
            for j = 1:nPop
                if particles(j).Cost < particles(i).Cost
                    % Calculate Distance
                    r = norm(particles(i).Position - particles(j).Position);
                    % Attractiveness
                    beta = beta0 * exp(-gamma * r^2);
                    % Update Position with FA
                    particles(i).Position = particles(i).Position ...
                        + beta * (particles(j).Position - particles(i).Position) ...
                        + alpha * (rand(1, nVar) - 0.5);
                    particles(i).Position = max(min(particles(i).Position, ub), lb);
                    particles(i).Cost = CostFunction(particles(i).Position);
                end
            end
        end
        
        % Update Global Best (PSO)
        [~, idx] = min([particles.pBestCost]);
        gBest = particles(idx);
        
        % Damping Parameters
        w = w * wDamp;
        alpha = alpha * alpha_damp;
    end
    
    BestSol = gBest;
end