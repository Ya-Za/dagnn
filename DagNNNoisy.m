classdef DagNNNoisy < handle
    %A framework for training a dag with noisy parameters
    
    properties
        % Properties
        % ----------
        % - noisy_config_dir: char vector
        %   Path of directory containing `template config` json files
        % - snr: double array (row vector)
        %   Each item is signal to noise ratio in dB.

        noisy_configs_dir;
        snr;
    end
    
    properties (Constant)
        FILENAME_PATTERN = 'snr_%g_bs_%g_lr_%g';
    end
    
    % Constructor
    methods
        function obj = DagNNNoisy(noisy_configs_dir)
            %Constructor
            %
            % Parameters
            % ----------
            % - noisy_config_dir: char vector
            %   Path of directory containing `template config` json files 
            
            if (~exist('noisy_configs_dir', 'var'))
                noisy_configs_dir = Path.NOISY_CONFIGS_DIR;
            end
            
            obj.noisy_configs_dir = noisy_configs_dir;
            
            obj.snr = [-1];
        end
    end
    
    % Run
    methods
        function run(obj)
            % RUN
            % parameters
            viz = DagNNViz();
            
            run('vl_setupnn.m');
            
            noisy_configs_filenames = obj.get_noisy_configs_filenames();
            for i = 1:numel(noisy_configs_filenames)
                noisy_configs_filename = noisy_configs_filenames{i};
                
                [~, name, ~] = fileparts(noisy_configs_filename);
                
                dt = datetime;
                dt.Format = 'dd-MMM-uuuu''.''HH-mm-ss';
                root_dir = fullfile(...
                    Path.RESULTS_DIR, ...
                    sprintf('%s.%s', name, dt) ...
                );
                % make dir
                if ~exist(root_dir, 'dir')
                    mkdir(root_dir);
                end
                
                for snr_value = obj.snr
                    % save `snr` to `info.mat`
                    info = struct();
                    info.snr = snr_value;
                    DagNNNoisy.saveToInfo(root_dir, info);
                    
                    % make db
                    db_filename = fullfile(root_dir, Path.DATA_FILENAME);
                    DagNNNoisy.make_db(...
                        noisy_configs_filename, ...
                        db_filename, ...
                        snr_value ...
                    );
                    
                    % make params
                    config = jsondecode(fileread(noisy_configs_filename));
                    
                    DagNNNoisy.make_params(...
                        noisy_configs_filename, ...
                        root_dir, ...
                        snr_value ...
                    );

                    % make config files
                    % todo: change `root dir` based on the following
                    % pattern
%                     config_filename = fullfile(...
%                         root_dir, ...
%                         sprintf(...
%                             ['config_' DagNNNoisy.FILENAME_PATTERN '.json'], ...
%                             snr_value, ...
%                             config.learning.batch_size, ...
%                             config.learning.learning_rate ...
%                         ) ...
%                     );
                    config_filename = fullfile(...
                            root_dir, ...
                            Path.CONFIG_FILENAME ...
                        );
%                     bak_dir = fullfile(...
%                         root_dir, ...
%                         sprintf(...
%                             ['bak_' DagNNNoisy.FILENAME_PATTERN], ...
%                             snr_value, ...
%                             config.learning.batch_size, ...
%                             config.learning.learning_rate ...
%                         ) ...
%                     );
                    bak_dir = root_dir;

                    params_initial_filename = fullfile(...
                        root_dir, ...
                        Path.PARAMS_INITIAL_FILENAME ...
                    );
                    DagNNNoisy.make_config(...
                        noisy_configs_filename, ...
                        db_filename, ...
                        params_initial_filename, ...
                        bak_dir, ...
                        config_filename ...
                    );

                    % run config files
                    DagNNNoisy.run_config(config_filename);

                    % todo: uncomment to plot figures
%                     % make images
%                     DagNNViz.plot_results(config_filename);
%                     
% %                     % copy net.svg
% %                     copyfile(...
% %                         fullfile(obj.noisy_configs_dir, [name, '.svg']), ...
% %                         fullfile(bak_dir, 'images', 'net.svg') ...
% %                     );
%                 
%                     % copy `index.html`
%                     copyfile(...
%                         Path.INDEX_HTML_FILENAME, ...
%                         bak_dir ...
%                     );
%                 
%                     % plot noisy/noiseless filters
%                     viz.output_dir = fullfile(bak_dir, 'images');
%                     viz.plot_noisy_params(...
%                         config_filename, ...
%                         config.data.params_filename, ...
%                         params_initial_filename, ...
%                         snr_value ...
%                     )
                end
            end
        end
        function filenames = get_noisy_configs_filenames(obj)
            % Get filenames of `noisy configs` from given directory
            %
            % Returns
            % -------
            % - filenames: cell array of char vectors
            %   `folder` + `name` of each `noisy configs` file
            
            listing = ...
                dir(fullfile(obj.noisy_configs_dir, '*.json'));
            
            filenames = arrayfun(...
                @(x) fullfile(x.folder, x.name), ...
                listing, ...
                'UniformOutput', false ...
            );
        end
    end
    
    methods (Static)
        % todo: make this method nonstatic
        function make_db(config_filename, db_filename, snr)
            % Make database based on dag (specivied by `config` file) 
            % and save it
            %
            % Parameters
            % ----------
            % - config_filename: char vector
            %   Path of dag config file
            % - db_filename: char vector
            %   Path of output database
            
            if exist(db_filename, 'file')
                return;
            end
            
            cnn = DagNNTrainer(config_filename);
            cnn.init();

            % db
            db.x = cnn.db.x;
            if snr == Inf
                db.y = cnn.db.y;
            else
                db.y = cnn.out(db.x);
            end
            
            % save
            save(...
                db_filename, ...
                '-struct', 'db' ...
            );
        end
        
        % todo: make this method nonstatic
        function make_params(config_filename, root_dir, snr)
            % Add noise to parameters of a dag and save it
            %
            % Parameters
            % ----------
            % - config_filename: char vector
            %   Path of dag config file
            % - params_filename: char vector
            %   Path of output dag parameters file
            % - snr: double
            %   Signal to noise ratio in dB
            
            params_initial_filename = fullfile(...
                root_dir, ...
                Path.PARAMS_INITIAL_FILENAME ...
            );
            if exist(params_initial_filename, 'file')
                return;
            end
            
            % net
            cnn = DagNNTrainer(config_filename);
            cnn.init();
            
            % params
            params = struct();
            for i = 1:length(cnn.net.params)
                params.(cnn.net.params(i).name) = cnn.net.params(i).value;
            end
            
            % save
            params_expected_filename = fullfile(...
                root_dir, ...
                Path.PARAMS_EXPECTED_FILENAME ...
            );
            save(...
                params_expected_filename, ...
                '-struct', 'params' ...
            );

            % add white Gaussian noise to signal
            fields = fieldnames(params);
            for i = 1 : length(fields)
                params.(fields{i}) = ...
                    awgn(params.(fields{i}), snr);
                    % awgn(params.(fields{i}), snr, 'measured');
            end

            % save
            save(...
                params_initial_filename, ...
                '-struct', 'params' ...
            );
            clear('params');
        end
        
        function make_config(...
                noisy_config_filename, ...
                db_filename, ...
                params_filename, ...
                bak_dir, ...
                config_filename ...
            )
            % Make a dag config file
            %
            % Parameters
            % ----------
            % - noisy_config_filename: char vector
            %   Path of noisy config file
            % - db_filename: char vector
            %   Path of database
            % - parame_filename: char vector
            %   Path of parameters
            % - bak_dir: char vector
            %   Path of backup directory
            % - config_filename: char vector
            %   Path of output config file
            
            if exist(config_filename, 'file')
                return;
            end
            
            % json
            % - decode
            config = jsondecode(fileread(noisy_config_filename));
            
            % - db_filename
            config.data.db_filename = db_filename;

            % - params_filename
            config.data.params_filename = params_filename;
            
            % - bak_dir
            config.data.bak_dir = bak_dir;
            
            % - encode and save
            file = fopen(config_filename, 'w');
            fprintf(file, '%s', jsonencode(config));
            fclose(file);
        end
        
        function run_config(config_filename)
            % Run a dag and plot `costs` and `diagraph`
            %
            % Parameters
            % ----------
            % - config_filename: char vector
            %   Path of config file for defining dag

            cnn = DagNNTrainer(config_filename);
            cnn.run();
        end
        
        function saveToInfo(rootDir, info)
            % Save additional information to `info.mat` file
            %
            % Parameters
            % ----------
            % - rootDir: cahr vector
            %   Path of root directory
            % - info: struct
            %   Must be added to `info.mat` file
            
            filename = fullfile(rootDir, Path.INFO_FILENAME);
            
            if exist(filename, 'file')
                % append
                save(filename, '-struct', 'info', '-append');
            else
                % create
                save(filename, '-struct', 'info');
            end
        end
        
        function main()
            % Main
            close('all');
            clear;
            clc;
            
            % parameters
            noisy = DagNNNoisy();
            noisy.run();
        end
    end
    
    % RMSE
    methods (Static)
        function rmse(configFilename)
            % Helper method for calling another functions
            %
            % Parameters
            % ----------
            % - configFilename: char vector
            %   Path of config file for defining dag
            
            % save `y_`
            DagNNNoisy.savePredictedOutput(configFilename);
            % save `indexes`
            DagNNNoisy.saveDBIndexes(configFilename);
        end
        
        function savePredictedOutput(configFilename)
            % Make predicted outputs and append the to the `db` as a `y_`
            %
            % Parameters
            % ----------
            % - configFilename: char vector
            %   Path of config file for defining dag
            
            % setup `matconvnet`
            run('vl_setupnn.m');
            
            % construct and init a dag
            cnn = DagNNTrainer(configFilename);
            cnn.init();
            cnn.load_best_val_epoch();
            
            % make predicted outputs
            y_ = cnn.out(cnn.db.x);
            
            % append predicted outputs to the current `db`
            save(cnn.config.data.db_filename, 'y_', '-append');
        end
        
        function saveDBIndexes(configFilename)
            % Append indexes of `train`, `val`, `test` data to the `db`
            %
            % Parameters
            % ----------
            % - configFilename: char vector
            %   Path of config file for defining dag
            
            % setup `matconvnet`
            run('vl_setupnn.m');
            
            % construct and init a dag
            cnn = DagNNTrainer(configFilename);
            cnn.init();
            
            % db indexes
            indexes = cnn.dbIndexes;
            
            % append predicted outputs to the current `db`
            save(cnn.config.data.db_filename, 'indexes', '-append');
        end
    end
    
end
