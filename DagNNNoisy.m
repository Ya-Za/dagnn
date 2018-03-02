classdef DagNNNoisy < handle
    %A framework for training a dag with noisy parameters
    
    properties
        % Properties
        % ----------
        % - base_props_dir: char vector
        %   Path of directory containing `base_props` json files
        % - index_html_path: char vector
        %   Path of `index.html` file
        % - snr: double array (row vector)
        %   Each item is signal to noise ratio in dB.

        base_props_dir = './data/ep20c11/noisy/base_props';
        index_html_path = './data/ep20c11/noisy/index.html';
        snr = [-10];
    end
    
    properties (Constant)
        FILENAME_PATTERN = 'snr_%g_bs_%g_lr_%g';
    end
    
    methods
        function filenames = get_base_props_filenames(obj)
            % Get filenames of `base_props` from given directory
            %
            % Returns
            % -------
            % - filenames: cell array of char vectors
            %   `folder` + `name` of each `base properties` file
            
            listing = ...
                dir(fullfile(obj.base_props_dir, '*.json'));
            
            filenames = arrayfun(...
                @(x) fullfile(x.folder, x.name), ...
                listing, ...
                'UniformOutput', false ...
            );
        end
        
        function run(obj)
            % RUN
            % parameters
            viz = DagNNViz();
            
            run('vl_setupnn.m');
            
            base_props_filenames = obj.get_base_props_filenames();
            for i = 1:numel(base_props_filenames)
                base_props_filename = base_props_filenames{i};
                
                [pathstr, ~, ~] = fileparts(obj.base_props_dir);
                [~, name, ~] = fileparts(base_props_filename);
                root_dir = fullfile(pathstr, 'results', name);
                % make dir
                if ~exist(root_dir, 'dir')
                    mkdir(root_dir);
                end
                
                % make db
                db_filename = fullfile(root_dir, 'db.mat');
                DagNNNoisy.make_db(...
                    base_props_filename, ...
                    db_filename ...
                );
                
                for snr_value = obj.snr
                    % make params
                    props = jsondecode(fileread(base_props_filename));
                    params_filename = fullfile(...
                        root_dir, ...
                        sprintf('params_snr_%g.mat', snr_value) ...
                    );
                    DagNNNoisy.make_params(...
                        base_props_filename, ...
                        params_filename, ...
                        snr_value ...
                    );

                    % make props files
                    props_filename = fullfile(...
                        root_dir, ...
                        sprintf(...
                            ['props_' DagNNNoisy.FILENAME_PATTERN '.json'], ...
                            snr_value, ...
                            props.learning.batch_size, ...
                            props.learning.learning_rate ...
                        ) ...
                    );
                    bak_dir = fullfile(...
                        root_dir, ...
                        sprintf(...
                            ['bak_' DagNNNoisy.FILENAME_PATTERN], ...
                            snr_value, ...
                            props.learning.batch_size, ...
                            props.learning.learning_rate ...
                        ) ...
                    );

                    DagNNNoisy.make_props(...
                        base_props_filename, ...
                        db_filename, ...
                        params_filename, ...
                        bak_dir, ...
                        props_filename ...
                    );

                    % run props files
                    DagNNNoisy.run_props(props_filename);
                    
                    % make images
                    DagNNViz.plot_results(props_filename);
                    
%                     % copy net.svg
%                     copyfile(...
%                         fullfile(obj.base_props_dir, [name, '.svg']), ...
%                         fullfile(bak_dir, 'images', 'net.svg') ...
%                     );
                
                    % copy index.html
                    copyfile(...
                        obj.index_html_path, ...
                        bak_dir ...
                    );
                
                    % plot noisy/noiseless filters
                    viz.output_dir = fullfile(bak_dir, 'images');
                    viz.plot_noisy_params(...
                        props_filename, ...
                        props.data.params_filename, ...
                        params_filename, ...
                        snr_value ...
                    )
                end
            end
        end
    end
    
    methods (Static)
        function make_db(props_filename, db_filename)
            % Make database based on dag (specivied by `props` file) 
            % and save it
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of dag properties file
            % - db_filename: char vector
            %   Path of output database
            
            if exist(db_filename, 'file')
                return;
            end
            
            cnn = DagNNTrainer(props_filename);
            cnn.init();

            % db
            db.x = cnn.db.x;
            db.y = cnn.out(db.x);
            
            % save
            save(...
                db_filename, ...
                '-struct', 'db' ...
            );
        end
        
        function make_params(props_filename, params_filename, snr)
            % Add noise to parameters of a dag and save it
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of dag properties file
            % - params_filename: char vector
            %   Path of output dag parameters file
            % - snr: double
            %   Signal to noise ratio in dB
            
            if exist(params_filename, 'file')
                return;
            end
            
            % net
            cnn = DagNNTrainer(props_filename);
            cnn.init();
            
            % load params
            params = load(cnn.props.data.params_filename);

            % add white Gaussian noise to signal
            fields = fieldnames(params);
            for i = 1 : length(fields)
                params.(fields{i}) = ...
                    awgn(params.(fields{i}), snr, 'measured');
            end

            % save
            save(...
                params_filename, ...
                '-struct', 'params' ...
            );
            clear('params');
        end
        
        function make_props(...
                base_props_filename, ...
                db_filename, ...
                params_filename, ...
                bak_dir, ...
                props_filename ...
            )
            % Make a dag properties file
            %
            % Parameters
            % ----------
            % - base_props_filename: char vector
            %   Path of basic properties file
            % - db_filename: char vector
            %   Path of database
            % - parame_filename: char vector
            %   Path of parameters
            % - bak_dir: char vector
            %   Path of backup directory
            % - props_filename: char vector
            %   Path of output properties file
            
            if exist(props_filename, 'file')
                return;
            end
            
            % json
            % - decode
            props = jsondecode(fileread(base_props_filename));
            
            % - db_filename
            props.data.db_filename = db_filename;

            % - params_filename
            props.data.params_filename = params_filename;
            
            % - bak_dir
            props.data.bak_dir = bak_dir;
            
            % - encode and save
            file = fopen(props_filename, 'w');
            fprintf(file, '%s', jsonencode(props));
            fclose(file);
        end
        
        function run_props(props_filename)
            % Run a dag and plot `costs` and `diagraph`
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of properties for defining dag

            cnn = DagNNTrainer(props_filename);
            cnn.run();
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
        function rmse(propsFilename)
            % Helper method for calling another functions
            %
            % Parameters
            % ----------
            % - propsFilename: char vector
            %   Path of properties for defining dag
            
            % save `y_`
            DagNNNoisy.savePredictedOutput(propsFilename);
            % save `indexes`
            DagNNNoisy.saveDBIndexes(propsFilename);
        end
        
        function savePredictedOutput(propsFilename)
            % Make predicted outputs and append the to the `db` as a `y_`
            %
            % Parameters
            % ----------
            % - propsFilename: char vector
            %   Path of properties for defining dag
            
            % setup `matconvnet`
            run('vl_setupnn.m');
            
            % construct and init a dag
            cnn = DagNNTrainer(propsFilename);
            cnn.init();
            cnn.load_best_val_epoch();
            
            % make predicted outputs
            y_ = cnn.out(cnn.db.x);
            
            % append predicted outputs to the current `db`
            save(cnn.props.data.db_filename, 'y_', '-append');
        end
        
        function saveDBIndexes(propsFilename)
            % Append indexes of `train`, `val`, `test` data to the `db`
            %
            % Parameters
            % ----------
            % - propsFilename: char vector
            %   Path of properties for defining dag
            
            % setup `matconvnet`
            run('vl_setupnn.m');
            
            % construct and init a dag
            cnn = DagNNTrainer(propsFilename);
            cnn.init();
            
            % db indexes
            indexes = cnn.dbIndexes;
            
            % append predicted outputs to the current `db`
            save(cnn.props.data.db_filename, 'indexes', '-append');
        end
    end
    
end
