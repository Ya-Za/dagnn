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
        % - INFO_FILENAME: char vector
        %   Additional information filename
        % - RESULTS_DIR: char vector
        %   Results directory
        % - BASE_CONFIGS_DIR: char vector
        %   Base Configs directory
        % - INDEX_HTML_FILENAME: char vector
        %   Path of `index.html` file
        
        EPOCHS_DIR = 'epochs';
        VIDEOS_DIR = 'vidoes';
        
        CONFIG_FILENAME = 'config.json';
        
        COSTS_FILENAME = 'costs.mat';
        DATA_FILENAME = 'data.mat';
        DATA_INDEXES_FILENAME = 'data-indexes.mat';
        PARAMS_EXPECTED_FILENAME = 'params-expected.mat';
        PARAMS_INITIAL_FILENAME = 'params-initial.mat';
        ELAPSED_TIMES_FILENAME = 'elapsed-times.mat';
        INFO_FILENAME = 'info.mat';
        
        RESULTS_DIR = './assets/results/';
        NOISY_CONFIGS_DIR = './assets/noisy-configs';
        INDEX_HTML_FILENAME = './assets/index.html';
    end
end