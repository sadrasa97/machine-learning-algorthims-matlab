clc
clear
close all
warning off
c
%% Create Random Solution

list = {'Truss 25','Frame 16','Truss 31','Frame 56'};
PromptString='Select a example';
idx = listdlg('PromptString',PromptString,'ListString',list);

n=1000;
DamageRatio=unifrnd(0.05,0.3,n,1);
isSymmetry=false;
Symmetry=[];

switch idx
    case 1
        nF=10;   % max 21
        DamageLocation=randi(25,[n 1]);
        NaturalFrequancy=zeros(n,nF);
    case 2
        nF=20;   % max 45
        DamageLocation=randi(16,[n 1]);
        NaturalFrequancy=zeros(n,nF);
        isSymmetry=true;
        Symmetry=[1 2 3 4; 5 6 7 8; 9 10 13 14; 11 12 15 16];
    case 3
        nF=20;   % max 24
        DamageLocation=randi(31,[n 1]);
        NaturalFrequancy=zeros(n,nF);
        %     isSymmetry=false;
        %     Symmetry=[1 31; 6 26; 11 21; 2 27; 7 22; 12 17; 5 30; 10 25; 15 20; 3 29; 8 24; 13 19; 4 28; 9 23; 14 18];
    case 4
        nF=10;   % max 220
        DamageLocation=randi(56,[n 1]);
        NaturalFrequancy=zeros(n,nF);
end

%% Find Solution by OPT
list = {'FA', 'GA', 'PSO', 'GWO', ...
    'PSO-GA', 'PSO-FA', 'PSO-GWO', ...
    'PSO-GA-FA', 'PSO-GA-GWO', 'PSO-GWO-FA'};
PromptString='Select an Algorithm';
indx = listdlg('PromptString',PromptString,'ListString',list);

nPop=25;
maxIt=200;
CostFunction=@(x) CostFcn(x);
nVar=2;
lb=[1 0];
ub=[100,0.3];

switch indx
    case 1  % FA (Firefly Algorithm)
        Sol = FA(CostFunction, nVar, maxIt, nPop, lb, ub);

    case 2  % GA (Genetic Algorithm)
        Sol = GA(CostFunction, nVar, maxIt, nPop, lb, ub);

    case 3  % PSO (Particle Swarm Optimization)
        Sol = PSO(CostFunction, nVar, lb, ub, nPop, maxIt);

    case 4  % GWO (Grey Wolf Optimizer)
        Sol = GWO(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 5  % PSO-GA
        Sol = PSO_GA(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 6  % PSO-FA
        Sol = PSO_FA(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 7  % PSO-GWO
        Sol = PSO_GWO(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 8  % PSO-GA-FA
        Sol = PSO_GA_FA(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 9  % PSO-GA-GWO
        Sol = PSO_GA_GWO(CostFunction, nVar, nPop, maxIt, lb, ub);

    case 10  % PSO-GWO-FA
        Sol = PSO_GWO_FA(CostFunction, nVar, nPop, maxIt, lb, ub);

    otherwise
        disp('This algorithm is not yet implemented or incorrect selection.');
        Sol = []; % Return an empty solution if an invalid selection is made
end

disp('Location :')
disp(num2str(round(Sol.Position(1))))

disp('Damage Ratio :')
disp(num2str(Sol.Position(2)))
