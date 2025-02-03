function bestSol = GA(CostFunction, nVar, maxIt, nPop, pc, TournamentSize, lb, ub, sigma, mu)
    % GA Parameters
    pm = mu;               % Mutation probability (mu from main script)
    mutationStep = sigma;  % Mutation step size (sigma from main script)
    gamma = 0.1;           % Crossover parameter
    
    % Initialize Population
    pop.Position = [];
    pop.Cost = [];
    pop = repmat(pop, nPop, 1);
    
    for i = 1:nPop
        pop(i).Position = lb + rand(1, nVar) .* (ub - lb);
        pop(i).Cost = CostFunction(pop(i).Position);
    end
    
    % Sort Population by Cost
    [~, sortIdx] = sort([pop.Cost]);
    pop = pop(sortIdx);

    bestSol = pop(1);  % Store the Best Solution

    % Main GA Loop
    for it = 1:maxIt
        newPop = pop;
        
        % Crossover
        for i = 1:2:nPop
            if rand < pc
                p1 = pop(randi(nPop)).Position;
                p2 = pop(randi(nPop)).Position;
                
                beta = (1 + 2 * gamma) * rand - gamma;
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2);
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2);
                
                newPop(i).Position = max(lb, min(ub, c1));
                newPop(i + 1).Position = max(lb, min(ub, c2));
                
                newPop(i).Cost = CostFunction(newPop(i).Position);
                newPop(i + 1).Cost = CostFunction(newPop(i + 1).Position);
            end
        end

        % Mutation
        for i = 1:nPop
            if rand < pm  % Use pm (set to mu from main script)
                newPop(i).Position = newPop(i).Position + mutationStep * (ub - lb) .* randn(1, nVar);
                newPop(i).Position = max(lb, min(ub, newPop(i).Position));
                newPop(i).Cost = CostFunction(newPop(i).Position);
            end
        end

        % Sort & Select Next Generation
        [~, sortIdx] = sort([newPop.Cost]);
        pop = newPop(sortIdx);

        % Update Best Solution
        if pop(1).Cost < bestSol.Cost
            bestSol = pop(1);
        end
    end
end