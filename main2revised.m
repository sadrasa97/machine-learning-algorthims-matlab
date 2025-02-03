clc
clear
close all
warning off
addpath FA\ GA\ GWO\ PSO\ ...
        PSO-GA\ PSO-FA\ PSO-GWO\ ...
        PSO-GA-FA\ PSO-GA-GWO\ PSO-GWO-FA\;
%% Create Random Solution

list = {'Truss 25','Frame 16','Truss 31','Frame 56'};
PromptString='Select a example';
idx = listdlg('PromptString',PromptString,'ListString',list);

DamageRatio=0.2;
DamageLocation=5;
nF=10;

switch idx
    case 1
        model=CreateModel1(DamageRatio,DamageLocation);
        NaturalFrequancy=TrussFEM(model,nF);
    case 2
        model=CreateModel2(DamageRatio,DamageLocation);
        NaturalFrequancy=FrameFEM(model,nF);
    case 3
        model=CreateModel3(DamageRatio,DamageLocation);
        NaturalFrequancy=TrussFEM(model,nF);
    case 4
        model=CreateModel4(DamageRatio,DamageLocation);
        NaturalFrequancy=FrameFEM(model,nF);
end

%% Find Solution by OPT
list = {'FA', 'GA', 'PSO', 'GWO', ...
    'PSO-GA', 'PSO-FA', 'PSO-GWO', ...
    'PSO-GA-FA', 'PSO-GA-GWO', 'PSO-GWO-FA'};
PromptString='Select a Algorithm';
indx = listdlg('PromptString',PromptString,'ListString',list);

nPop=25;
maxIt=200;
CostFunction=@(x) CostFcn(x,idx,NaturalFrequancy);
nVar=2;
ne=numel(model.A);
lb=[1 0];
ub=[ne,0.3];



switch indx
    case 1 % FA
        gamma = 1;            % Light Absorption Coefficient
        beta0 = 2;            % Attraction Coefficient Base Value
        alpha = 0.2;          % Mutation Coefficient
        alpha_damp = 0.99;    % Mutation Coefficient Damping Ratio
        delta = 0.05*(lb-ub); % Uniform Mutation Range
        m = 2;
        Sol = FA(CostFunction, nVar, maxIt, nPop, gamma, beta0, alpha, alpha_damp, m, lb, ub);

    case 2 % GA
        pc = 0.5;
        TournamentSize = 5;
        mu = 0.4;
        sigma = 0.1;
        Sol = GA(CostFunction, nVar, maxIt, nPop, pc, TournamentSize, lb, ub, sigma, mu);

    case 3 % PSO
        w = 1.5;     
        c1 = 3;       
        c2 = 5;       
        wDamp = 0.99; 
        Sol = PSO(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp);

    case 4 % GWO
        Sol = GWO(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 5 % PSO-GA
        w = 1.5;
        c1 = 2;
        c2 = 2;
        wDamp = 0.99;
        
        pc = 0.8;
        mu = 0.1;
        sigma = 0.15;
        
        Sol = PSO_GA(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, pc, mu, sigma);

    case 6 % PSO-FA
        w = 1.2;
        c1 = 1.7;
        c2 = 1.7;
        wDamp = 0.99;
        
        gamma = 0.5;
        beta0 = 1;
        alpha = 0.3;
        alpha_damp = 0.98;
        
        Sol = PSO_FA(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, gamma, beta0, alpha, alpha_damp);

    case 7 % PSO-GWO
        w = 1.0;
        c1 = 2.0;
        c2 = 2.0;
        wDamp = 0.95;
        
        a = 2;      % Exploration Parameter
        a_damp = 1; % Damping Ratio
        
        Sol = PSO_GWO(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, a, a_damp);

    case 8 % PSO-GA-FA
        w = 1.2;
        c1 = 2.0;
        c2 = 2.0;
        wDamp = 0.97;
        
        pc = 0.7;
        mu = 0.2;
        sigma = 0.1;
        
        gamma = 0.6;
        beta0 = 1.5;
        alpha = 0.25;
        alpha_damp = 0.99;
        
        Sol = PSO_GA_FA(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, pc, mu, sigma, gamma, beta0, alpha, alpha_damp);

    case 9 % PSO-GA-GWO
        w = 1.0;
        c1 = 1.8;
        c2 = 1.8;
        wDamp = 0.98;
        
        pc = 0.75;
        mu = 0.15;
        sigma = 0.2;
        
        a = 2;
        a_damp = 0.95;
        
        Sol = PSO_GA_GWO(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, pc, mu, sigma, a, a_damp);

    case 10 % PSO-GWO-FA
        w = 1.3;
        c1 = 2.0;
        c2 = 2.0;
        wDamp = 0.96;
        
        a = 2;
        a_damp = 0.9;
        
        gamma = 0.7;
        beta0 = 1.2;
        alpha = 0.2;
        alpha_damp = 0.99;
        
        Sol = PSO_GWO_FA(CostFunction, nVar, lb, ub, nPop, maxIt, w, c1, c2, wDamp, a, a_damp, gamma, beta0, alpha, alpha_damp);
end

disp('Location :')
disp(num2str(round(Sol.Position(1))))

disp('Damage Ratio :')
disp(num2str(Sol.Position(2)))

