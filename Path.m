classdef Path < handle
    %Path of specific fils and folders
    
    properties (Constant)
        % Properties
        % ----------
        % - EPOCHS_DIR: char vector
        %   Epochs directory
        % - VIDEOS_DIR: char vector
        %   Videos directory
        % - CONFIG_FILENAME: char vector
        %   Config filename
        % - COSTS_FILENAME: char vector
        %   Costs filename
        % - DATA_FILENAME: char vector
        %   Data filename
        % - DATA_INDEXES_FILENAME: char vector
        %   Data indexes filename
        % - PARAMS_EXPECTED_FILENAME: char vector
        %   Expected parameters filename
        % - PARAMS_INITIAL_FILENAME: char vector
        %   Initial parameters filename
        
        EPOCHS_DIR = 'epochs';
        VIDEOS_DIR = 'vidoes';
        
        CONFIG_FILENAME = 'config.json';
        
        COSTS_FILENAME = 'costs.mat';
        DATA_FILENAME = 'data.mat';
        DATA_INDEXES_FILENAME = 'data-indexes.mat';
        PARAMS_EXPECTED_FILENAME = 'params-expected';
        PARAMS_INITIAL_FILENAME = 'params-initial';
    end
end