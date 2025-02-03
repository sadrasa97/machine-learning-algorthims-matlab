function BestSol = PSO_GA_FA(CostFunction, nVar, lb, ub, nPop, maxIt, ...
                             w, c1, c2, wDamp, pc, mu, sigma, gamma, beta0, alpha, alpha_damp)
    % Parameters:
    % - PSO: w, c1, c2, wDamp
    % - GA: pc (crossover prob), mu (mutation prob), sigma (mutation step)
    % - FA: gamma, beta0, alpha, alpha_damp

    % Initialize Population
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

        % GA Crossover
        for i = 1:2:nPop
            if rand < pc
                % Select Parents
                parent1 = particles(randi(nPop)).pBest;
                parent2 = particles(randi(nPop)).pBest;
                
                % Uniform Crossover
                mask = rand(1, nVar) < 0.5;
                child1 = parent1 .* mask + parent2 .* (~mask);
                child2 = parent2 .* mask + parent1 .* (~mask);
                
                particles(i).Position = child1;
                particles(i+1).Position = child2;
            end
        end

        % GA Mutation
        for i = 1:nPop
            if rand < mu
                particles(i).Position = particles(i).Position + sigma * randn(1, nVar);
                particles(i).Position = max(min(particles(i).Position, ub), lb);
                particles(i).Cost = CostFunction(particles(i).Position);
            end
        end

        % FA Attraction
        for i = 1:nPop
            for j = 1:nPop
                if particles(j).Cost < particles(i).Cost
                    r = norm(particles(i).Position - particles(j).Position);
                    beta = beta0 * exp(-gamma * r^2);
                    particles(i).Position = particles(i).Position ...
                        + beta * (particles(j).Position - particles(i).Position) ...
                        + alpha * (rand(1, nVar) - 0.5);
                    particles(i).Position = max(min(particles(i).Position, ub), lb);
                    particles(i).Cost = CostFunction(particles(i).Position);
                end
            end
        end

        % Update Global Best
        [~, idx] = min([particles.pBestCost]);
        gBest = particles(idx);
        
        % Damping
        w = w * wDamp;
        alpha = alpha * alpha_damp;
    end

    BestSol = gBest;
end