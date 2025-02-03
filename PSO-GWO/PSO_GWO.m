function BestSol = PSO_GWO(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, a, a_damp)
    % Parameters:
    % - PSO Parameters: w, c1, c2, wDamp
    % - GWO Parameters: a (exploration parameter), a_damp

    % Initialize Particles/Wolves
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
        
        % GWO Hunting Mechanism
        a_val = 2 * a * (1 - (it/maxIt));  % Linearly decrease a
        for i = 1:nPop
            % Update positions using Alpha, Beta, Delta
            X1 = alpha.Position - a_val * rand(1, nVar) .* abs(rand(1, nVar) .* alpha.Position - particles(i).Position);
            X2 = beta.Position - a_val * rand(1, nVar) .* abs(rand(1, nVar) .* beta.Position - particles(i).Position);
            X3 = delta.Position - a_val * rand(1, nVar) .* abs(rand(1, nVar) .* delta.Position - particles(i).Position);
            
            % Average of Alpha, Beta, Delta positions
            particles(i).Position = (X1 + X2 + X3) / 3;
            particles(i).Position = max(min(particles(i).Position, ub), lb);
            particles(i).Cost = CostFunction(particles(i).Position);
        end
        
        % Update Alpha, Beta, Delta
        [~, sortedIdx] = sort([particles.Cost]);
        alpha = particles(sortedIdx(1));
        beta = particles(sortedIdx(2));
        delta = particles(sortedIdx(3));
        
        % Damping Parameters
        w = w * wDamp;
        a = a * a_damp;
    end
    
    BestSol = alpha;
end