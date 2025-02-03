function BestSol = PSO_GA(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, pc, mu, sigma)

    particles = repmat(struct('Position', [], 'Velocity', [], 'Cost', [], 'pBest', [], 'pBestCost', []), nPop, 1);
    for i = 1:nPop
        particles(i).Position = lb + rand(1, nVar) .* (ub - lb);
        particles(i).Velocity = zeros(1, nVar);
        particles(i).Cost = CostFunction(particles(i).Position);
        particles(i).pBest = particles(i).Position;
        particles(i).pBestCost = particles(i).Cost;
    end
    
    [~, idx] = min([particles.Cost]);
    gBest = particles(idx);

    for it = 1:maxIt
        for i = 1:nPop
            r1 = rand(1, nVar);
            r2 = rand(1, nVar);
            particles(i).Velocity = w * particles(i).Velocity ...
                + c1 * r1 .* (particles(i).pBest - particles(i).Position) ...
                + c2 * r2 .* (gBest.Position - particles(i).Position);
            
            particles(i).Position = particles(i).Position + particles(i).Velocity;
            particles(i).Position = max(min(particles(i).Position, ub), lb);
            
            particles(i).Cost = CostFunction(particles(i).Position);
            
            if particles(i).Cost < particles(i).pBestCost
                particles(i).pBest = particles(i).Position;
                particles(i).pBestCost = particles(i).Cost;
            end
        end
        
        [~, idx] = min([particles.pBestCost]);
        gBest = particles(idx);
        
        w = w * wDamp;
        
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
        
        for i = 1:nPop
            if rand < mu
                particles(i).Position = particles(i).Position + sigma * randn(1, nVar);
                particles(i).Position = max(min(particles(i).Position, ub), lb);
                particles(i).Cost = CostFunction(particles(i).Position);
            end
        end
    end
    
    BestSol = gBest;
end