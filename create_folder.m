folders = {'PSO', 'GA', 'GWO', 'FA', 'SVM', 'ANN', 'ANFIS', ...
           'SVM-PSO', 'PSO-GA', 'PSO-FA', 'PSO-GWO', ...
           'SVM-PSO-GWO', 'SVM-PSO-FA', 'SVM-PSO-GA', ...
           'ANN-PSO-GA', 'ANN-PSO-FA', 'ANN-PSO-GWO', ...
           'PSO-GA-FA', 'PSO-GA-GWO', 'PSO-GWO-FA', ...
           'SVM-PSO-GWO-FA', 'SVM-PSO-GA-FA', 'SVM-PSO-GA-GWO', ...
           'ANN-PSO-GWO-FA', 'ANN-PSO-GA-GWO', 'ANN-PSO-GA-FA'};

for i = 1:length(folders)
    folderName = folders{i};
    if ~exist(folderName, 'dir')
        mkdir(folderName);
        fprintf('Folder "%s" created.\n', folderName);
    else
        fprintf('Folder "%s" already exists.\n', folderName);
    end
end
