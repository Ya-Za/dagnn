classdef Neda < handle
    %Neda
    %   Detailed explanation goes here
    
    properties (Constant)
        exp_filename = 'C:\Users\Yasin\Dropbox\Neda\retina\CNN\Data\uniform field stimuli\exp 20\mat files\makedata_repeated_selected_ep20c11';
        dt_sec = 0.001;
        l_sec = 1.000;
        d_sec = 0.100;
        refresh_rate = 13;
        output_dir = './data/ep20c11';
    end
    
    methods (Static)
        function output = resize(input, refresh_rate)
            % RESIZE resizes 'input' based on new 'refresh-rate' sampling-
            % rate. each batch replaces by its mean
            %
            % Parameters
            % ----------
            % - input: double vector
            %   input vector
            % - refresh_rate: int
            %   new sampling rate
            %
            % Returns
            % -------
            % - output: double vector
            %   resized output vector
            %
            % Examples
            % --------
            % 1. 
            %   >>> input = [1, 2, 3, 4, 5];
            %   >>> refresh_rate = 2;
            %   >>> Neda.resize(input, refresh_rate)
            %   [1.5, 3.5]
            
            % remove residual elements of 'input', because input must be
            % dividable by 'refresh-rate'
            input_length = ...
                refresh_rate * ...
                floor(length(input) / refresh_rate);
            input = input(1:input_length);
            % make output
            output =  mean(reshape(input, refresh_rate, []));
        end
        
        function output = downsample(input, refresh_rate)
            % DOWNSAMPLE resamples 'input' based on new 'refresh-rate'
            % sampling-rate. each batch replaces by its mean
            %
            % Parameters
            % ----------
            % - input: double vector
            %   input vector
            % - refresh_rate: int
            %   new sampling rate
            %
            % Returns
            % -------
            % - output: double vector
            %   resized output vector
            %
            % Examples
            % --------
            % 1. 
            %   >>> input = [1, 2, 3, 4, 5];
            %   >>> refresh_rate = 2;
            %   >>> Neda.downsample(input, refresh_rate)
            %   [1, 3]
            
            % remove residual elements of 'input', because input must be
            % dividable by 'refresh-rate'
            input_length = ...
                refresh_rate * ...
                floor(length(input) / refresh_rate);
            input = input(1:input_length);
            % make output
            output = downsample(input, refresh_rate);
        end
        
        function save_db(exp_filename, dt_sec, l_sec, d_sec, refresh_rate, output_dir)
            % SAVE_DB makes database 'db' from saved 'exp_filename'
            % experiment and save it in 'output_dir/db.mat' file.
            % db = struct('x', cell array, 'y', cell array)
            %
            % Parameters
            % ----------
            % - exp_filename: char vector
            %   filename of saved experiment
            % - dt_sec: double
            %   time resolution in seconds
            % - l_sec: double
            %   length of sub-signal in seconds
            % - d_sec: double
            %   delta between two sub-signlas in secondss
            % - refresh_rate: int
            %   refresh-rate of the monitor
            % - output-dir: char vector
            %   path of output directory
            
            % default values
            if nargin == 0
                exp_filename = Neda.exp_filename;
                dt_sec = Neda.dt_sec;
                l_sec = Neda.l_sec;
                d_sec = Neda.d_sec;
                refresh_rate = Neda.refresh_rate;
                output_dir = Neda.output_dir;
            end
            
            % stim
            % - read
            vstim_rep = getfield(load(exp_filename), 'vstim_rep');
            stim = vstim_rep(1, 1:end-1);
            % - mean remove
            stim = stim - mean(stim(:));
            
            % resp
            % - read
            resp = getfield(load(exp_filename), 'PSTH1_y_s');
            
            % divide
            % - stim
            stims = DataUtils.divide_timeseries(...
                stim, ...
                dt_sec, ...
                l_sec, ...
                d_sec ...
            );
            % - resp
            resps = DataUtils.divide_timeseries(...
                resp, ...
                dt_sec, ...
                l_sec, ...
                d_sec ...
            );
        
            % resize
            % - stims
            resized_stims = [];
            for i = 1 : size(stims, 1)
                resized_stims(i, :) = Neda.resize(stims(i, :), refresh_rate);
            end
            % - resps
            resized_resps = [];
            for i = 1 : size(resps, 1)
                resized_resps(i, :) = Neda.resize(resps(i, :), refresh_rate);
            end
            
            % db
            % - make
            db.x = num2cell(resized_stims', 1)';
            db.y = num2cell(resized_resps', 1)';
            % - save
            save(...
                fullfile(output_dir, 'db.mat'), ...
                '-struct', ...
            'db');
            
            % save data
            save(...
                fullfile(output_dir, 'data.mat'), ...
                'stim', ...
                'resp', ...
                'stims', ...
                'resps' ...
            );
        end
        
        function save_params(exp_filename, refresh_rate, output_dir)
            % SAVE_PARAMS makes parameters 'params' from saved 'exp_filename'
            % experiment and save it in 'output_dir/params.mat' directory.
            % params = struct(...
            %           'w_B', double vector, ...
            %           'w_A', double vector, ...
            %           'w_G', double vector, ...
            %           'b_B', double, ...
            %           'b_A', double, ...
            %           'b_G', double ...
            % )
            %
            % Parameters
            % ----------
            % - exp_filename: char vector
            %   filename of saved experiment
            % - refresh_rate: int
            %   refresh-rate of the monitor
            % - output-dir: char vector
            %   path of output directory
            
            % default values
            if nargin == 0
                exp_filename = Neda.exp_filename;
                refresh_rate = Neda.refresh_rate;
                output_dir = Neda.output_dir;
            end
            
            % read filters
            % - FG
            FG = getfield(load(exp_filename), 'FG');
            % - FA
            FA = getfield(load(exp_filename), 'FA');
            
            % downsample
            % - FG
            FG = Neda.downsample(FG, refresh_rate);
            % - FA
            FA = Neda.downsample(FA, refresh_rate);
            
            % filters
            % - make
            params.w_B = FG';
            params.w_A = FA';
            params.w_G = FG';
            params.b_B = 0;
            params.b_A = 0;
            params.b_G = 0;
            % - save
            save(fullfile(output_dir, 'params.mat'), '-struct', 'params');
        end
    end
    
end

