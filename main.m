clc
clear
close all
warning off
%addpath SVM\ANN\SVM-PSO\ ANN-PSO\SVM-PSO-GA\ SVM-PSO-GWO\SVM-PSO-FA\SVM-PSO-GA-FA\ SVM-PSO-GA-GWO\SVM-PSO-GWO-FA

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

for i=1:n
    switch idx
        case 1
            model=CreateModel1(DamageRatio(i),DamageLocation(i));
            NaturalFrequancy(i,:)=TrussFEM(model,nF);
        case 2
            model=CreateModel2(DamageRatio(i),DamageLocation(i));
            NaturalFrequancy(i,:)=FrameFEM(model,nF);
        case 3
            model=CreateModel3(DamageRatio(i),DamageLocation(i));
            NaturalFrequancy(i,:)=TrussFEM(model,nF);
        case 4
            model=CreateModel4(DamageRatio(i),DamageLocation(i));
            NaturalFrequancy(i,:)=FrameFEM(model,nF);
    end
end
if isSymmetry
    for i=1:size(Symmetry,1)
        for j=1:size(Symmetry,2)-1
            DamageLocation(DamageLocation==Symmetry(i,j+1))=Symmetry(i,1);
        end
    end
end

%% Fitting
list = {'SVM','ANN','ANFIS','SVM-PSO','ANN-PSO','SVM-PSO-GA', 'SVM-PSO-GWO', 'SVM-PSO-FA', 'SVM-PSO-GA-FA', 'SVM-PSO-GA-GWO', 'SVM-PSO-GWO-FA'};
PromptString='Select a Method';
indx = listdlg('PromptString',PromptString,'ListString',list);

TrainRatio=0.7;
TestRatio=0.15;
MaxIterations=15;
SwarmSize=8;

hiddenLayerSize=[10 10 10];

switch indx
    case 1
        RegressionSol=RegressionSVM(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio);
        ClassificationSol = ClassifierSVM(NaturalFrequancy',DamageLocation',hiddenLayerSize,TrainRatio,TestRatio,isSymmetry,Symmetry);
    case 2
        RegressionSol=regressionANN(NaturalFrequancy,DamageRatio,hiddenLayerSize,TrainRatio,TestRatio);
        ClassificationSol=ClassifierANN(NaturalFrequancy',DamageLocation',hiddenLayerSize,TrainRatio,TestRatio,isSymmetry,Symmetry);
    case 3
        RegressionSol=RegressionANFIS(NaturalFrequancy,DamageRatio,TrainRatio, TestRatio);
        ClassificationSol=ClassifierANFIS(NaturalFrequancy,DamageLocation,TrainRatio,0.3,isSymmetry,Symmetry);
    case 4
        MaxIterations = 1000;       
        SwarmSize = 8;            
        maxIterSVR = 1000;         
        learningRate = 0.001;      

        RegressionSol = RegressionSVMPSO(...
            NaturalFrequancy, DamageRatio, ...
            TrainRatio, TestRatio, ...
            MaxIterations, SwarmSize, ...
            maxIterSVR, learningRate ...
        );

        ClassificationSol = ClassifierSVMPSO(...
            NaturalFrequancy, DamageLocation, ...
            TrainRatio, TestRatio, MaxIterations, ...
            SwarmSize, isSymmetry, Symmetry ...
        );
    case 5
        RegressionSol=RegressionANNPSO(NaturalFrequancy,DamageRatio,TrainRatio,TestRatio,MaxIterations,SwarmSize);
        ClassificationSol=ClassifierANNPSO(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);
    case 6
        PopulationSize = 50;  
        MaxGen = 100;  

        RegressionSol = RegressionSVMPSOGA(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio, MaxIterations, SwarmSize, PopulationSize, MaxGen);

        ClassificationSol=ClassifierSVMPSOGA(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);    
    case 7
        RegressionSol=RegressionSVMPSOGWO(NaturalFrequancy,DamageRatio,TrainRatio,TestRatio,MaxIterations,SwarmSize);
        ClassificationSol=ClassifierSVMPSOGWO(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);
    case 8
        RegressionSol=RegressionSVMPSOFA(NaturalFrequancy,DamageRatio,TrainRatio,TestRatio,MaxIterations,SwarmSize);
        ClassificationSol=ClassifierSVMPSOFA(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);
    case 9
        RegressionSol=RegressionSVMPSOGAFA(NaturalFrequancy,DamageRatio,TrainRatio,TestRatio,MaxIterations,SwarmSize);
        ClassificationSol=ClassifierSVMPSOGAFA(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);
    case 10
        RegressionSol=RegressionSVMPSOGAGWO(NaturalFrequancy,DamageRatio,TrainRatio,TestRatio,MaxIterations,SwarmSize);
        ClassificationSol=ClassifierSVMPSOGAGWO(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);
    case 11
        RegressionSol=RegressionSVMPSOGWOFA(NaturalFrequancy,DamageRatio,TrainRatio,TestRatio,MaxIterations,SwarmSize);
        ClassificationSol=ClassifierSVMPSOGWOFA(NaturalFrequancy,DamageLocation,TrainRatio,TestRatio,MaxIterations,SwarmSize,isSymmetry,Symmetry);
end

figure;
m = confusionmat(ClassificationSol.TestTrue, ClassificationSol.TestPredictions);

if exist('confusionchart', 'file')
    confusionchart(m);
else
    imagesc(m);
    colorbar;
    title('Confusion Matrix');
    xlabel('Predicted Class');
    ylabel('True Class');
    axis square;
end