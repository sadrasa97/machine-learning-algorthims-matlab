function cost = CostFcn(x, idx, NaturalFrequancy)
    % Extract and clamp/round damage location
    ne = 25;  % Replace 25 with actual number of elements for each case
    damageLocation = round(x(1));
    damageLocation = max(1, min(ne, damageLocation));  % Ensure 1 ? index ? ne
    
    damageRatio = x(2);
    % Recalculate natural frequencies based on the damage parameters
    switch idx
        case 1  % Truss 25
            modelDamaged = CreateModel1(damageRatio, damageLocation);
            NaturalFrequancyDamaged = TrussFEM(modelDamaged, length(NaturalFrequancy));
        case 2  % Frame 16
            modelDamaged = CreateModel2(damageRatio, damageLocation);
            NaturalFrequancyDamaged = FrameFEM(modelDamaged, length(NaturalFrequancy));
        case 3  % Truss 31
            modelDamaged = CreateModel3(damageRatio, damageLocation);
            NaturalFrequancyDamaged = TrussFEM(modelDamaged, length(NaturalFrequancy));
        case 4  % Frame 56
            modelDamaged = CreateModel4(damageRatio, damageLocation);
            NaturalFrequancyDamaged = FrameFEM(modelDamaged, length(NaturalFrequancy));
    end
    
    % Compute cost as the difference between measured and actual frequencies
    cost = sum((NaturalFrequancyDamaged - NaturalFrequancy).^2);
end