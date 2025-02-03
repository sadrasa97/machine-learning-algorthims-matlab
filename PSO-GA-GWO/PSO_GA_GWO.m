function BestSol = PSO_GA_GWO(CostFunction, nVar, lb, ub, nPop, maxIt, ...
                              w, c1, c2, wDamp, pc, mu, sigma, a, a_damp)
    % Parameters:
    % - PSO: w, c1, c2, wDamp
    % - GA: pc, mu, sigma
    % - GWO: a (exploration parameter), a_damp

    % Initialize Population
    particles = repmat(struct('Position', [], 'Velocity', [], 'Cost', [], 'pBest', [], 'pBestCost', []), nPop, 1);
    for i = 1:nPop
        particles(i).Position = lb + rand(1, nVar) .* (ub - lb);
        particles(i).Velocity = zeros(1, nVar);
        particles(i).Cost = CostFunction(particles(i).Position);
        particles(i).pBest = particles(i).Position;
        particles(i).pBestCost = particles(i).Cost;
    end

    % Initialize Alpha, Beta, Delta Wolves (GWO)
    [~, sortedIdx] = sort([particles.Cost]);
    alpha = particles(sortedIdx(1));
    beta = particles(sortedIdx(2));
    delta = particles(sortedIdx(3));

    % Main Loop
    for it = 1:maxIt
        % PSO Velocity Update
        for i = 1:nPop
            r1 = rand(1, nVar);
            r2 = rand(1, nVar);
            particles(i).Velocity = w * particles(i).Velocity ...
                + c1 * r1 .* (particles(i).pBest - particles(i).Position) ...
                + c2 * r2 .* (alpha.Position - particles(i).Position);
            
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
                parent1 = particles(randi(nPop)).pBest;
                parent2 = particles(randi(nPop)).pBest;
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

        % GWO Hunting
        a_val = 2 * a * (1 - (it/maxIt));  % Decrease a linearly
        for i = 1:nPop
            X1 = alpha.Position - a_val * rand(1, nVar) .* abs(alpha.Position - particles(i).Position);
            X2 = beta.Position - a_val * rand(1, nVar) .* abs(beta.Position - particles(i).Position);
            X3 = delta.Position - a_val * rand(1, nVar) .* abs(delta.Position - particles(i).Position);
            particles(i).Position = (X1 + X2 + X3) / 3;
            particles(i).Position = max(min(particles(i).Position, ub), lb);
            particles(i).Cost = CostFunction(particles(i).Position);
        end

        % Update Alpha, Beta, Delta
        [~, sortedIdx] = sort([particles.Cost]);
        alpha = particles(sortedIdx(1));
        beta = particles(sortedIdx(2));
        delta = particles(sortedIdx(3));
        
        % Damping
        w = w * wDamp;
        a = a * a_damp;
    end

    BestSol = alpha;
end