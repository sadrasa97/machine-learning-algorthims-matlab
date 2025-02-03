function [TrainData, TestData, TrainTarget, TestTarget] = splitData(NaturalFrequancy, DamageRatio, TrainRatio, TestRatio)
    % Determine the number of samples
    N = size(NaturalFrequancy, 1);
    TrainSize = round(TrainRatio * N);
    
    % Generate random indices for training and testing
    indices = randperm(N);
    TrainIdx = indices(1:TrainSize);
    TestIdx = indices(TrainSize+1:end);
    
    % Split Data
    TrainData = NaturalFrequancy(TrainIdx, :);
    TestData = NaturalFrequancy(TestIdx, :);
    TrainTarget = DamageRatio(TrainIdx, :);
    TestTarget = DamageRatio(TestIdx, :);
end

