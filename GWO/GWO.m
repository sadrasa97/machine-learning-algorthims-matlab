function Sol = GWO(CostFunction, nVar, nPop, maxIt, lb, ub)
    % Initialize wolves
    Wolves = rand(nPop, nVar) .* (ub - lb) + lb;
    Costs = arrayfun(@(i) CostFunction(Wolves(i, :)), 1:nPop);
    [~, sortedIdx] = sort(Costs);
    Alpha = Wolves(sortedIdx(1), :);
    Beta = Wolves(sortedIdx(2), :);
    Delta = Wolves(sortedIdx(3), :);
    
    for it = 1:maxIt
        a = 2 - it * (2 / maxIt);
        for i = 1:nPop
            A1 = 2 * a * rand(1, nVar) - a;
            C1 = 2 * rand(1, nVar);
            D_alpha = abs(C1 .* Alpha - Wolves(i, :));
            X1 = Alpha - A1 .* D_alpha;
            
            A2 = 2 * a * rand(1, nVar) - a;
            C2 = 2 * rand(1, nVar);
            D_beta = abs(C2 .* Beta - Wolves(i, :));
            X2 = Beta - A2 .* D_beta;
            
            A3 = 2 * a * rand(1, nVar) - a;
            C3 = 2 * rand(1, nVar);
            D_delta = abs(C3 .* Delta - Wolves(i, :));
            X3 = Delta - A3 .* D_delta;
            
            Wolves(i, :) = (X1 + X2 + X3) / 3;
            Wolves(i, :) = max(min(Wolves(i, :), ub), lb);
            
            newCost = CostFunction(Wolves(i, :));
            if newCost < Costs(i)
                Costs(i) = newCost;
            end
        end
        
        [~, sortedIdx] = sort(Costs);
        Alpha = Wolves(sortedIdx(1), :);
        Beta = Wolves(sortedIdx(2), :);
        Delta = Wolves(sortedIdx(3), :);
    end
    
    Sol.Position = Alpha;
    Sol.Cost = min(Costs);
end
