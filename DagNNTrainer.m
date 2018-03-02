classdef DagNNTrainer < handle
    %Trainer for DagNN
    properties
        % - props: struct base on `dagnntrainer_schema.json`
        %   Properties of cnn contains configuration of 'data', 'net'
        %   and 'learning' parameters
        % - db: struct('x', cell array, 'y', cell array)
        %   Database
        % - current_epoch : int
        %   Current epoch
        % - net: DagNN
        %   Dag convolutional neural network
        % - data: struct(...
        %           'train', struct('x', cell array, 'y', cell array), ...
        %           'val', struct('x', cell array, 'y', cell array), ...
        %           'test', struct('x', cell array, 'y', cell array) ...
        %         )
        %   Contains 'train', 'val' and 'test' data
        % - dbIndexes: struct(...
        %           'train', int vector, ...
        %           'val', int vector, ...
        %           'test', int vector ...
        %         )
        %   Contains 'train', 'val' and 'test' indexes
        % - costs: stuct(...
        %           'train', double array, ...
        %           'val', double array, ...
        %           'test', double array ...
        %          )
        %   History of `costs` for `training`, `validation` and `testing`
        %   data-sets
        % - elapsed_times: double array
        %   Array of elased times
        props
        db
        current_epoch
        net
        data
        dbIndexes
        costs
        elapsed_times
    end
    
    properties (Constant)
        % - props_dir: char vector
        %   Path of properties json files
        % - has_bias: logical
        %   True if `dagnn.Conv` has bias
        % - blocks: struct
        %   Structure of `name` and `handler` of `dagnn` blocks
        
        props_dir = './data/props';
        has_bias = false;
        % todo: must be documented
        format_spec = struct(...
            'change_db_y', '-changed_y.mat', ...
            'noisy_params', '-snr_%d.mat' ...
        );
        blocks = struct(...
            'conv', @dagnn.Conv, ...
            'norm', @dagnn.BatchNorm2, ... % @dagnn.NormOverall
            'relu', @dagnn.ReLU, ...
            'logsig', @dagnn.Sigmoid, ...
            'tansig', @dagnn.TanSigmoid, ...
            'neg', @dagnn.Neg, ... 
            'minus', @dagnn.Minus, ...
            'times', @dagnn.Times, ...
            'sum', @dagnn.Sum, ...
            'quadcost', @dagnn.PDist ...
        );
    end
    
    methods
        function obj = DagNNTrainer(props_filename)
            %Constructor
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of configuration json file
            
            % print 'Load: ...'
            [~, filename, ext] = fileparts(props_filename);
            DagNNViz.print_title(...
                sprintf('Load: "%s\" file', [filename, ext])...
            );
            
            obj.init_props(props_filename);
        end
        
        % todo: handle this in `init_net` method
        function init_props(obj, filename)
            % Read `props` from the configuration json file and
            % refine it such as convert column-vector to row-vector and
            % null to {}
            %
            % Parameters
            % ----------
            % - filename: char vector
            %   Path of configuration json file
            
            % decode json
            obj.props = jsondecode(fileread(filename));
            
            % net (column-vector -> row-vector)
            % - vars
            %   - input
            obj.props.net.vars.input.size = obj.props.net.vars.input.size';
            
            %   - output
            obj.props.net.vars.output.size = obj.props.net.vars.output.size';
            
            % - params
            for i = 1:length(obj.props.net.params)
                obj.props.net.params(i).size = obj.props.net.params(i).size';
            end
            
            % - layers (column-vector -> row-vector and null -> {})
            for i = 1:length(obj.props.net.layers)
                % - inputs
                if isempty(obj.props.net.layers(i).inputs)
                    obj.props.net.layers(i).inputs = {};
                else
                    obj.props.net.layers(i).inputs = obj.props.net.layers(i).inputs';
                end
                % - outputs
                if isempty(obj.props.net.layers(i).outputs)
                    obj.props.net.layers(i).outputs = {};
                else
                    obj.props.net.layers(i).outputs = obj.props.net.layers(i).outputs';
                end
                
                % - params
                if isempty(obj.props.net.layers(i).params)
                    obj.props.net.layers(i).params = {};
                else
                    obj.props.net.layers(i).params = obj.props.net.layers(i).params';
                end
            end
        end
        
        function init_db(obj)
            % Initialze 'db' from the 'db_filename'
            
            % db
            % - load
            obj.db = load(obj.props.data.db_filename);
            % - standardize
            obj.standardize_db();
            % - resize
            obj.resize_db();
        end
        
        function standardize_db(obj)
            % Make db zero-mean and unit-variance
            
            % db
            % - x
            if obj.props.learning.standardize_x
                for i = 1 : length(obj.db.x)
                    obj.db.x{i} = normalize(obj.db.x{i});
                end
            end
            % - y
            if obj.props.learning.standardize_y
                for i = 1 : length(obj.db.y)
                    obj.db.y{i} = normalize(obj.db.y{i});
                end
            end
            
            % Local functions
            function x = normalize(x, eps)
                % Normalize vector `x` (zero mean, unit variance)

                % default values
                if (~exist('eps', 'var'))
                    eps = 1e-6;
                end

                mu = mean(x(:));

                sigma = std(x(:));
                if sigma < eps
                    sigma = 1;
                end

                x = (x - mu) ./ sigma;
            end
        end
        
        function resize_db(obj)
            % Resize 'db.x' and 'db.y'
            
            % resize
            % - db.x
            input_size = obj.props.net.vars.input.size;
            for i = 1 : length(obj.db.x)
                obj.db.x{i} = DataUtils.resize(obj.db.x{i}, input_size);
            end
            % - db.y
            output_size = obj.props.net.vars.output.size;
            for i = 1 : length(obj.db.y)
                obj.db.y{i} = DataUtils.resize(obj.db.y{i}, output_size);
            end
        end
        
        function init_bak_dir(obj)
            % Initialize `bak` directory with `bak_dir`
            
            if ~exist(obj.props.data.bak_dir, 'dir')
                mkdir(obj.props.data.bak_dir);
            end
        end
        
        function init_current_epoch(obj)
            % Initialize `current_epoch` based on last
            % saved epoch in the `bak` directory
            
            list = dir(fullfile(obj.props.data.bak_dir, 'epoch_*.mat'));
            % todo: [\d] -> \d
            tokens = regexp({list.name}, 'epoch_([\d]+).mat', 'tokens');
            epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens);
            obj.current_epoch = max(epoch);
        end
        
        function init_net(obj)
            % Initialze `net`
            
            % if there is no saved epoch file in 'bak' directory
            if isempty(obj.current_epoch)
                obj.current_epoch = 1;
                
                % define object
                % todo: Add `dagnn` to path
                obj.net = dagnn.DagNN();
                % obj.net.conserveMemory = false;
                
                % add layers
                layers = obj.props.net.layers;
                for layer_index = 1:length(layers)
                    layer = layers(layer_index);
                    
                    % type = subtype1 + subtype2 + ...
                    sub_types = cellfun(...
                        @strtrim, ...
                        strsplit(layer.type, '+'), ...
                        'UniformOutput', false ...
                    );
                    number_of_sub_types = length(sub_types);
                    
                    if number_of_sub_types == 1
                        obj.net.addLayer(...
                            layer.name, DagNNTrainer.blocks.(layer.type)(), ... %todo why we must use `()` at the end of this line?! and is there any better way to do that?
                            layer.inputs, ...
                            layer.outputs, ...
                            layer.params ...
                        );
                    else
                        % sub inputs
                        sub_inputs = cell(1, number_of_sub_types);
                        sub_inputs{1} = layer.inputs;
                        for sub_input_index = 2:number_of_sub_types
                            sub_inputs{sub_input_index} = ...
                                [layer.outputs{1}, '_', num2str(sub_input_index - 1)];
                        end
                        
                        % sub outpus
                        sub_outputs = {sub_inputs{2:end}, layer.outputs{1}};
                        
                        % sub params
                        if isempty(layer.params)
                            layer.params = cell(1, number_of_sub_types);
                        end
                        sub_params = cell(1, number_of_sub_types);
                        for sub_param_index = 1:number_of_sub_types
                            if isempty(layer.params{sub_param_index})
                                sub_params{sub_param_index} = {};
                            else
                                sub_params{sub_param_index} = ...
                                    layer.params{sub_param_index}';
                            end
                        end
                        
                        % add layers
                        for sub_type_index = 1:number_of_sub_types
                            sub_type = sub_types{sub_type_index};
                            sub_input = sub_inputs{sub_type_index};
                            sub_output = sub_outputs{sub_type_index};
                            sub_param = sub_params{sub_type_index};
                            
                            obj.net.addLayer(...
                                [layer.name, '_',  sub_type], DagNNTrainer.blocks.(sub_type)(), ...
                                sub_input, ...
                                sub_output, ...
                                sub_param ...
                            );
                        end
                    end
                end
                
                % init params
                obj.init_params();
                
                % set `size` and `hasBias` property of `Conv` blocks
                for layer_index = 1:length(obj.net.layers)
                    if startsWith(...
                            class(obj.net.layers(layer_index).block), ...
                            'dagnn.Conv' ...
                            )
                        param_name = obj.net.layers(layer_index).params{1};
                        sub_param_index = obj.net.getParamIndex(param_name);
                        param_size = size(obj.net.params(sub_param_index).value);
                        
                        % `size`
                        obj.net.layers(layer_index).block.size = param_size;
                        
                        % `hasBias`
                        if length(obj.net.layers(layer_index).params) == 1
                            obj.net.layers(layer_index).block.hasBias = false;
                        end
                    end
                end
                
                % save first epoch in `bak` directory
                obj.save_current_epoch();
            else
                % load last saved epoch file in `bak` directory
                obj.load_current_epoch();
            end
        end
        
        function init_data(obj)
            % Divide `db` based on `train_val_test_ratios` and
            % initializes 'data'
            
            % number of samples
            n = min(length(obj.db.x), length(obj.db.y));
            
            % ratios
            % todo: must normalize ratios -> ratio/sum_of_rations
            % - train
            ratios.train = obj.props.learning.train_val_test_ratios(1);
            % - test
            ratios.val = obj.props.learning.train_val_test_ratios(2);
            
            % shuffle db
            if exist(obj.get_db_indexes_filename(), 'file')
                indexes = obj.load_db_indexes();
            else
                indexes = randperm(n);
                obj.save_db_indexes(indexes);
            end
            
            % end index
            % - train
            end_index.train = floor(ratios.train * n);
            % - val
            end_index.val = floor((ratios.train + ratios.val) * n);
            % - test
            end_index.test = n;
            
            % data
            % - train
            obj.dbIndexes.train = sort(indexes(1:end_index.train));
            %   - x
            obj.data.train.x = obj.db.x(obj.dbIndexes.train);
            %   - y
            obj.data.train.y = obj.db.y(obj.dbIndexes.train);
            
            % - val
            obj.dbIndexes.val = sort(indexes(end_index.train + 1:end_index.val));
            %   - x
            obj.data.val.x = obj.db.x(obj.dbIndexes.val);
            %   - y
            obj.data.val.y = obj.db.y(obj.dbIndexes.val);
            
            % - test
            obj.dbIndexes.test = sort(indexes(end_index.val + 1:end_index.test));
            %   - x
            obj.data.test.x = obj.db.x(obj.dbIndexes.test);
            %   - y
            obj.data.test.y = obj.db.y(obj.dbIndexes.test);
        end
        
        function init_params(obj)
            % Initialize obj.net.params from `params_filename`
            
            params = obj.props.net.params;
            weights = load(obj.props.data.params_filename);
%             disp('Must be changed');
%             weights.w_G = randn(10, 1);
            for i = 1:length(params)
                name = params(i).name;
                size = params(i).size;
                index = obj.net.getParamIndex(name);
                
                obj.net.params(index).value = ...
                    DataUtils.resize(weights.(name), size);
            end
        end
        
        function init_meta(obj)
            % Set obj.net.meta
            
%             obj.net.meta = struct(...
%                 'learning_rate', obj.props.learning.learning_rate, ...
%                 'batch_size', obj.props.learning.batch_size ...
%             );

            obj.net.meta.learning = obj.props.learning;
        end
        
        function cost = get_cost(obj, x, y)
            % Compute mean cost of net based on input `x` and
            % expected-output `y`
            %
            % Parameters
            % ----------
            % - x: cell array
            %   Input
            % - y: cell array
            %   Expected-output
            %
            % Returns
            % -------
            % - cost: double
            %   Cost of net based on given inputs
            
            n = numel(x);
            cost = 0;
            for i = 1:n
                obj.net.eval({...
                    obj.props.net.vars.input.name, x{i}, ...
                    obj.props.net.vars.expected_output.name, y{i} ...
                });
                
                cost = cost + obj.net.vars(...
                    obj.net.getVarIndex(obj.props.net.vars.cost.name) ...
                ).value;
            end
            
            cost = cost / n;
        end
        
        function train_cost = get_train_cost(obj)
            % Get cost of training data
            %
            % Returns
            % -------
            % - train_cost: double
            %   Cost of net for `training` data-set
            
            train_cost = ...
                obj.get_cost(obj.data.train.x, obj.data.train.y);
        end
        
        function val_cost = get_val_cost(obj)
            % Get cost of validation data
            %
            % Returns
            % -------
            % - val_cost: double
            %   Cost of net for `validation` data-set
            
            val_cost = ...
                obj.get_cost(obj.data.val.x, obj.data.val.y);
        end
        
        function test_cost = get_test_cost(obj)
            % Get cost of test data
            %
            % Returns
            % -------
            % - test_cost: double
            %   Cost of net for `testing` data-set
            
            test_cost = ...
                obj.get_cost(obj.data.test.x, obj.data.test.y);
        end
        
        function init_costs(obj)
            % Initialize `obj.costs`
            
            % if `costs.mat` file exists in `bak` directory
            if exist(obj.get_costs_filename(), 'file')
                obj.load_costs();
                obj.costs.train = ...
                    obj.costs.train(1:obj.current_epoch);
                obj.costs.val = ...
                    obj.costs.val(1:obj.current_epoch);
                obj.costs.test = ...
                    obj.costs.test(1:obj.current_epoch);
            else
                % costs
                % - train
                obj.costs.train(1) = obj.get_train_cost();
                
                % - costs
                obj.costs.val(1) = obj.get_val_cost();
                
                % - costs
                obj.costs.test(1) = obj.get_test_cost();
                
                % save
                obj.save_costs();
            end
        end
        
        function init_elapsed_times(obj)
            % Initialize `elapsed_times`
            
            % if `elapsed_times` file exists in `bak` directory
            if exist(obj.get_elapsed_times_filename(), 'file')
                obj.load_elapsed_times();
                obj.elapsed_times = ...
                    obj.elapsed_times(1:obj.current_epoch);
            else
                obj.elapsed_times(1) = 0;
                obj.save_elapsed_times();
            end
        end
        
        function init(obj)
            % Initialize properties
            
            % db
            obj.init_db();
            
            % backup directory
            obj.init_bak_dir()
            
            % current epoch
            obj.init_current_epoch()
            
            % net
            obj.init_net();
            
            % - meta
            obj.init_meta();
            
            % data
            obj.init_data();
            
            % costs
            obj.init_costs();
            
            % elapsed times
            obj.init_elapsed_times();
            
            % bak_dir == 'must_be_removed'
            if strcmp(obj.props.data.bak_dir, 'must_be_removed')
                rmdir(obj.props.data.bak_dir, 's');
            end
        end
        
        function y = out(obj, x)
            % Compute `estimated-outputs` of network based on given
            % `inputs`
            %
            % Parameters
            % ----------
            % - x: cell array
            %   Input
            % - y: cell array
            %   Actual output
            
            n = numel(x);
            y = cell(n, 1);
            for i = 1:n
                obj.net.eval({...
                    obj.props.net.vars.input.name, x{i} ...
                });
                
                y{i} = obj.net.vars(...
                    obj.net.getVarIndex(obj.props.net.vars.output.name) ...
                ).value;
            end
        end
        
        function load_best_val_epoch(obj)
            % Load best validation performance among saved epochs
            
            % update current-epoch
            [~, obj.current_epoch] = min(obj.costs.val);
            % init-net
            % todo: efficient way to change the net based on just `currnt
            % epoch`
            obj.init_net();
        end
        
        function print_epoch_progress(obj)
            % Print progress, after each batch
            %
            % Examples
            % --------
            % 1.
            %   >>> print_epoch_progress()
            %   --------------------------------
            %   Epoch:	...
            %   Costs:	[..., ..., ...]
            %   Time:	... s
            %   --------------------------------
            
            DagNNViz.print_dashline();
            fprintf('Epoch:\t%d\n', obj.current_epoch);
            fprintf('Costs:\t[%.3f, %.3f, %.3f]\n', ...
                obj.costs.train(end), ...
                obj.costs.val(end), ...
                obj.costs.test(end) ...
                );
            fprintf('Time:\t%f s\n', ...
                obj.elapsed_times(obj.current_epoch));
            DagNNViz.print_dashline();
        end
        
        % todo: Must be removed or change to `real time` plot
        function plot_costs(obj)
            % Plot `costs` over time
            
            epochs = 1:length(obj.costs.train);
            epochs = epochs - 1; % start from zero (0, 1, 2, ...)
            
            figure(...
                'Name', 'CNN - Costs [Training, Validation, Test]', ...
                'NumberTitle', 'off', ...
                'Units', 'normalized', ...
                'OuterPosition', [0.25, 0.25, 0.5, 0.5] ...
                );
            
            % costs
            % - train
            plot(epochs, obj.costs.train, 'LineWidth', 2, 'Color', 'blue');
            set(gca, 'YScale', 'log');
            hold('on');
            % - validation
            plot(epochs, obj.costs.val, 'LineWidth', 2, 'Color', 'green');
            % - test
            plot(epochs, obj.costs.test, 'LineWidth', 2, 'Color', 'red');
            
            % minimum validation error
            % - circle
            [~, index_min_val_cost] = min(obj.costs.val);
            circle_x = index_min_val_cost - 1;
            circle_y = obj.costs.val(index_min_val_cost);
            dark_green = [0.1, 0.8, 0.1];
            scatter(circle_x, circle_y, ...
                'MarkerEdgeColor', dark_green, ...
                'SizeData', 300, ...
                'LineWidth', 2 ...
                );
            
            % - cross lines
            h_ax = gca;
            %   - horizontal line
            line(...
                h_ax.XLim, ...
                [circle_y, circle_y], ...
                'Color', dark_green, ...
                'LineStyle', ':', ...
                'LineWidth', 1.5 ...
                );
            %   - vertical line
            line(...
                [circle_x, circle_x], ...
                h_ax.YLim, ...
                'Color', dark_green, ...
                'LineStyle', ':', ...
                'LineWidth', 1.5 ...
                );
            
            hold('off');
            
            % labels
            xlabel('Epoch');
            ylabel('Mean Squared Error (mse)');
            
            % title
            title(...
                sprintf('Minimum Validation Error is %.3f at Epoch: %d', ...
                obj.costs.val(index_min_val_cost), ...
                index_min_val_cost - 1 ...
                ) ...
                );
            
            % legend
            legend(...
                sprintf('Training (%.3f)', obj.costs.train(index_min_val_cost)), ...
                sprintf('Validation (%.3f)', obj.costs.val(index_min_val_cost)), ...
                sprintf('Test (%.3f)', obj.costs.test(index_min_val_cost)), ...
                'Best' ...
                );
            
            % grid
            grid('on');
            grid('minor');
        end
        
        function filename = get_current_epoch_filename(obj)
            % Get path of `current epoch`
            % saved file in `bak` directory
            %
            % Returns
            % -------
            % - filename: char vector
            
            filename = fullfile(...
                obj.props.data.bak_dir, ...
                sprintf('epoch_%d', obj.current_epoch) ...
            );
        end
        
        function save_current_epoch(obj)
            % Save `net` of current-epoch in `bak`
            % directory
            
            net_struct = obj.net.saveobj();
            save(...
                obj.get_current_epoch_filename(), ...
                '-struct', 'net_struct' ...
            ) ;
            
            clear('net_struct');
        end
        
        function load_current_epoch(obj)
            % Load `net` of current-epoch from `bak`
            % directory
            
            net_struct = load(...
                obj.get_current_epoch_filename() ...
            );
            
            obj.net = dagnn.DagNN.loadobj(net_struct) ;
            clear('net_struct');
        end
        
        function filename = get_costs_filename(obj)
            % Return path of `costs.mat` saved file in
            % `bak` directory
            
            filename = fullfile(...
                obj.props.data.bak_dir, ...
                'costs.mat' ...
            );
        end
        
        % todo: save `costs` in `meta` data of each epoch
        function save_costs(obj)
            % Save `costs.mat` in `bak` directory
            
            costs = obj.costs;
            
            save(...
                obj.get_costs_filename(), ...
                '-struct', ...
                'costs' ...
            );
            
            clear('costs');
        end
        
        function load_costs(obj)
            % Load `costs.mat` from `bak` directory
            
            obj.costs = load(obj.get_costs_filename());
        end
        
        function filename = get_db_indexes_filename(obj)
            % Return path of `db_indexes.mat`
            % saved file in `bak` directory
            
            filename = fullfile(...
                obj.props.data.bak_dir, ...
                'db_indexes.mat' ...
            );
        end
        
        function save_db_indexes(obj, indexes)
            % Save `db_indexes.mat` in `bak` directory
            
            db_indexes = indexes;
            save(...
                obj.get_db_indexes_filename(), ...
                'db_indexes' ...
            );
        end
        
        function db_indexes = load_db_indexes(obj)
            % Loads `db_indexes.mat` from `bak` directory
            
            db_indexes = getfield(...
                load(obj.get_db_indexes_filename()), ...
                'db_indexes' ...
            );
        end
        
        function filename = get_elapsed_times_filename(obj)
            % Return path of `elapsed_times.mat`
            % saved file in `bak` directory
            
            filename = fullfile(...
                obj.props.data.bak_dir, ...
                'elapsed_times.mat' ...
            );
        end
        
        % todo: save `elapsed times` in `meta` data of each epoch
        function save_elapsed_times(obj)
            % Save `elapsed_times` in `bak` directory
            
            elapsed_times = obj.elapsed_times;
            save(...
                obj.get_elapsed_times_filename(), ...
                'elapsed_times' ...
            );
            
            clear('elapsed_times');
        end
        
        function load_elapsed_times(obj)
            % Load `elapsed_times.mat` from `bak` directory
            
            obj.elapsed_times = getfield(...
                load(obj.get_elapsed_times_filename()), ...
                'elapsed_times' ...
            );
        end
        
        function save(obj, filename)
            % Save the `DagNNTrainer` object
            
            save(filename, 'obj');
        end
        
        %todo: split to `db`, `params`
        function make_random_data_old(obj, number_of_samples, generator)
            % Make random `db` and `params` files
            %
            % Parameters
            % ----------
            % - number_of_samples : int
            %   number of training data
            % - generator : handle function (default is @rand)
            %   generator function such as `rand`, `randn`, ...
            
            % default generator
            if ~exist('generator', 'var')
                generator = @rand;
            end
            
            % db
            db.x = cell(number_of_samples, 1);
            db.y = cell(number_of_samples, 1);
            
            % - x, y
            input_size = obj.props.net.vars.input.size;
            output_size = obj.props.net.vars.output.size;
            for i = 1:number_of_samples
                db.x{i} = generator(input_size);
                db.y{i} = generator(output_size);
            end
            
            % - save
            % todo: save with `-struct` option
            save(obj.props.data.db_filename, 'db');
            clear('db');
            
            % params
            params = obj.props.net.params;
            % - weights
            weights = struct();
            for i = 1 : length(params)
                weights.(params(i).name) = generator(params(i).size);
            end
            % - save
            save(obj.props.data.params_filename, '-struct', 'weights');
            clear('weights');
        end
        
        function make_db_with_changed_y(obj)
            % Generate `db.y` based on given 
            % `db.x` and `params` file
            
            % db filename
            db_filename = [...
                obj.props.data.db_filename, ...
                DagNNTrainer.format_spec.change_db_y ...
            ];
        
            if exist(db_filename, 'file')
                return
            end
            
            % db
            db.x = obj.db.x;
            db.y = obj.out(db.x);
            
            % save
            save(...
                db_filename, ...
                '-struct', 'db' ...
            );
            clear('db');
        end
        
        function make_noisy_params(obj, snr)
            % Makes noisy `params` with given signal to nooise ratio in dB
            %
            % Parameters
            % ----------
            % - snr: double
            %   The scalar snr specifies the signal-to-noise ratio per 
            %   sample, in dB
            
            % params filename
            params_filename = [...
                obj.props.data.params_filename, ...
                sprintf(DagNNTrainer.format_spec.noisy_params, snr) ...
            ];
        
            if exist(params_filename, 'file')
                return
            end
            
            % load
            params = load(obj.props.data.params_filename);
            
            % add white Gaussian noise to signal
            for field = fieldnames(params)
                params.(char(field)) = ...
                    awgn(params.(char(field)), snr, 'measured');
            end
            
            % save
            save(...
                params_filename, ...
                '-struct', 'params' ...
            );
            clear('params');
        end
        
        function run(obj)
            % Run the learing process contains `forward`, `backward`
            % and `update` steps
            
            % init net
            obj.init();
            
            % print epoch progress (last saved epoch)
            obj.print_epoch_progress()
            
            obj.current_epoch = obj.current_epoch + 1;
            
            % epoch number that network has minimum cost on validation data
            [~, index_min_val_cost] = min(obj.costs.val);
            
            n = length(obj.data.train.x);
            batch_size = obj.props.learning.batch_size;
            
            % epoch loop
            while obj.current_epoch <= obj.props.learning.number_of_epochs + 1
                begin_time = cputime();
                % shuffle train data
                permuted_indexes = randperm(n);
                
                % batch loop
                for start_index = 1:batch_size:n
                    end_index = start_index + batch_size - 1;
                    if end_index > n
                        end_index = n;
                    end
                    
                    indexes = permuted_indexes(start_index:end_index);
                    % make batch data
                    % - x
                    input = ...
                        DagNNTrainer.cell_array_to_tensor(...
                        obj.data.train.x(indexes) ...
                    );
                    % - y
                    expected_output = ...
                        DagNNTrainer.cell_array_to_tensor(...
                        obj.data.train.y(indexes) ...
                    );
                    
                    % forward, backward step
                    obj.net.eval(...
                        {...
                            obj.props.net.vars.input.name, input, ...
                            obj.props.net.vars.expected_output.name, expected_output
                        }, ...
                        {...
                            obj.props.net.vars.cost.name, 1 ...
                        } ...
                    );
                    
                    % update step
                    for param_index = 1:length(obj.net.params)
                        obj.net.params(param_index).value = ...
                            obj.net.params(param_index).value - ...
                            obj.props.learning.learning_rate * obj.net.params(param_index).der;
                    end
                    
                    % print samples progress
                    fprintf('Samples:\t%d-%d/%d\n', start_index, end_index, n);
                end
                
                % elapsed times
                obj.elapsed_times(end + 1) = cputime() - begin_time;
                % costs
                % - train
                obj.costs.train(end + 1) = obj.get_train_cost();
                % - val
                obj.costs.val(end + 1) = obj.get_val_cost();
                % - test
                obj.costs.test(end + 1) = obj.get_test_cost();
                
                % no imporovement in number_of_val_fails steps
                if obj.costs.val(end) < obj.costs.val(index_min_val_cost)
                    index_min_val_cost = length(obj.costs.val);
                end
                
                if (length(obj.costs.val) - index_min_val_cost) >= ...
                        obj.props.learning.number_of_val_fails
                    break;
                end
                
                % print epoch progress
                obj.print_epoch_progress()
                
                % save
                % - costs
                obj.save_costs();
                % - elapsed times
                obj.save_elapsed_times();
                % - net
                obj.save_current_epoch();
                
                % increament current epoch
                obj.current_epoch = obj.current_epoch + 1;
            end 
        end
    end
    
    methods (Static)
        function tensor = cell_array_to_tensor(cell_array)
            % Convert cell array to multi-dimensional array
            %
            % Parameters
            % ----------
            % - cell_array: cell_array
            %   Input cell array
            
            tensor_size = horzcat(...
                size(cell_array{1}), ...
                [1, length(cell_array)] ...
            );
            
            indexes = cell(1, length(tensor_size));
            for i = 1:length(tensor_size)
                indexes{i} = 1:tensor_size(i);
            end
            
            tensor = zeros(tensor_size);
            for i = 1:length(cell_array)
                indexes{end} = i;
                tensor(indexes{:}) = cell_array{i};
            end
        end
        
        function obj = load(filename)
            % Load `DagNNTrainer` from file
            
            obj = load(filename);
            obj = obj.(char(fieldnames(obj)));
        end
        
        function test1()
            % setup `matconvnet`
            run('vl_setupnn.m');
            
            % `props` dir
            props_dir = DagNNTrainer.props_dir;
            % properties filenames
            props_filenames = ...
                dir(fullfile(props_dir, '*.json'));
            props_filenames = {props_filenames.name};
            
            % net
            for i = 1 : length(props_filenames)
                % props-filename
                props_filename = fullfile(props_dir, props_filenames{i});
                % - define
                cnn = DagNNTrainer(props_filename);
                % - run
                tic();
                cnn.run();
                toc();
                
                % - plot net
                DagNNTrainer.plot_digraph(props_filename);
            end
        end
        
        function test
            % Test `DagNNTrainer` class
            suite = testsuite('./tests/TestDagNNTrainer.m');
            suite.run();
        end
    end 
end
