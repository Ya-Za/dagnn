%% Results
close('all');
clear();
clc();

addpath('./libs');
root = './assets/results';
%%
folders = dir(root);
folders = folders([folders.isdir]);
folders = folders(3:end)';

for folder = folders
    disp(folder.name);
    
    path = fullfile(folder.folder, folder.name);
    viz = Viz(path);
    
    viz.plotNet();
    
	viz.plotData(1:50);
	viz.boxplotData();
    viz.boardplotData();
    
	viz.plotCosts();
    % viz.plotErrors(@(u, v) corr(u, v));
    viz.plotCorr();
    
    viz.plotParameters();
    
    viz.plotExpectedActualOutputs();
    viz.plotExpectedActualOutputsOverall();
    
    % viz.playFilterVideo();
    
    viz.saveData();
    
    close('all');
end