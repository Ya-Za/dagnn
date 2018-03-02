classdef DagNNViz < handle
    %Visualize a 'DagNN'
    
    properties
        % Properties
        % -------------------
        % - bak_dir: char vector
        %   Path of `backup` directory which contains `costs`, `db_indexes`, 
        %   `elapsed_times`, `epochs` and also contains `summary`
        %   direcotry.
        % - data_dir: char vector
        %   Path of `data` directory which contains `db.mat` and
        %   `params.mat`
        % - output_dir: char vector
        %   Path of output directory
        % - formattype: char vector
        %   File format such as `epsc`, `pdf`, `svg`, `png` or ...
        % - showtitle: logical (default: true)
        %   If `showtitle` is `false` then plots don't have any title
        % - round_digits: int
        %   Minimum Number of digits to the right of the decimal point
        
        bak_dir = 'E:\Documents\University\3. PhD\MSU\Neda\codes\Retina\nn\convolutional_neural_network\cnn\data\ep20c11\fig4.2\bak_200_0.0001';
        data_dir = 'E:\Documents\University\3. PhD\MSU\Neda\codes\Retina\nn\convolutional_neural_network\cnn\data\ep20c11';
        output_dir = '';
        formattype = 'svg';
        showtitle = true;
        round_digits = 3;
    end
    
    % Unused
    methods (Static)
        
    end
    
    % Utils
    methods
        function title(obj, varargin)
            % Add or not `title` of current axis
            %
            % Parameters
            % ----------
            % - text: char vector
            %   Text of title
            if obj.showtitle
                title(varargin{:});
            end
        end
        
        function suptitle(obj, text)
            % Add or not `suptitle` of current figure
            %
            % Parameters
            % ----------
            % - text: char vector
            %   Text of title
            if obj.showtitle
                suptitle(text);
            end
        end
        
        function round_digits = get_round_digits(obj, x_min, x_max)
            % Get first `rounddigits` start from `0` that distinguish
            % between `x_min` and `x_max`
            %
            % Parameters
            % ----------
            % - x_min: double
            %   Begin of interval
            % - x_max: double
            %   End of interval
            %
            % Returns
            % -------
            % - round_digits: int
            %   Number of digits to the right of the decimal point
            
            round_digits = obj.round_digits;
            x_min_rounded = round(x_min, round_digits);
            x_max_rounded = round(x_max, round_digits);
            
            while x_min_rounded == x_max_rounded
                round_digits = round_digits + 1;
                x_min_rounded = round(x_min, round_digits);
                x_max_rounded = round(x_max, round_digits);
            end
        end
        
        function saveas(obj, filename)
            % Save curret figure as `fielname`
            %
            % Parameters
            % ----------
            % - filename: char vector
            %   Path of file which must be saved
            
            saveas(...
                gcf, ...
                fullfile(obj.output_dir, filename), ...
                obj.formattype ...
            );
        end
        
        function twoticks(obj, x, axis_name)
            % Show just `min` and `max` of ticks
            %
            % Parameters
            % ----------
            % - x: double vector
            %   Input values
            % - axis_name: cahr vector
            %   'XTick' or 'YTick' 
            
            x_min = min(x);
            x_max = max(x);
            rounddigits = obj.get_round_digits(x_min, x_max);
            set(gca, ...
                axis_name, [...
                    round(x_min, rounddigits), ...
                    round(x_max, rounddigits) ...
                ] ...
            );
        end
    end
    methods (Static)
        function h = figure(name)
            % Create `full screen` figure
            %
            % Parameters
            % ----------
            % - name: char vector
            %   Name of figure
            %
            % Return
            % - h: matlab.ui.Figure
            h = figure(...
                'Name', name, ...
                'NumberTitle', 'off', ...
                'Units', 'normalized', ...
                'OuterPosition', [0, 0, 1, 1] ...
            );
        end
        
        function hideticks()
            % Hide `XTick` and `YTick` of current axis
            set(gca, ...
                'XTick', [], ...
                'YTick', [], ...
                'Box', 'off' ...
            );
        end
        
        function pm = populationmean(x)
            % Return `mean` of population
            % Parameters
            % ----------
            % - x: cell array
            %   input data {x1, x2, ...}
            %
            % Return
            % ------
            % - pm: double vector
            
            % number of samples
            N = length(x);
            
            % population mean
            pm = x{1};
            for i = 2 : N
                pm = pm + x{i};
            end
            
            pm = pm / N;
        end
        
        function param_history = get_param_history(bak_dir, param_name)
            % Get history of a `param_name` prameter 
            % based on saved epochs in `bak_dir` directory
            %
            % Parameters
            % ----------
            % - bak_dir: char vector
            %   Path of directory of saved epochs
            % - param_name: char vector
            %   Name of target parameter
            %
            % Returns
            % -------
            % - param_history : cell array
            %   History of param values
            
            % name-list of saved 'epoch' files
            filenames = dir(fullfile(bak_dir, 'epoch_*.mat'));
            filenames = {filenames.name};
            
            % number of epochs
            N = length(filenames);
            
            % get index of param-name
            params = getfield(...
                load(fullfile(bak_dir, filenames{1})), ...
                'params' ...
            );
            param_index = cellfun(...
                @(x) strcmp(x, param_name), ...
                {params.name} ...
            );
        
            if ~any(param_index)
                param_history = {};
                return
            end
        
            % param-hsitory
            param_history = cell(N, 1);
            for i = 1 : N
                 params = getfield(...
                    load(fullfile(bak_dir, filenames{i})), ...
                    'params' ...
                ); 
                param_history{i} = params(param_index).value;
            end
        end
        
        function ylimits = get_ylimits(x)
            % Get ylimits that is suitable for all samples of
            % 'x' data
            %
            % Parameter
            % ---------
            % - x : cell array
            %   Input data
            %
            % Returns
            % -------
            % - ylimits : [double, double]
            %   Ylimtis of all samples of 'x'
            
            ylimits = [
                min(cellfun(@(s) min(s), x)), ...
                max(cellfun(@(s) max(s), x))
            ];
        end
        
        function save_estimated_outputs(props_filename)
            % Save estimated-outputs in 'bak_dir/y_.mat'
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of configuration json file
            
            % run 'vl_setupnn.m' file
            run('vl_setupnn.m');

            % cnn
            cnn = DagNNTrainer(props_filename);
            cnn.init();
            
            % load best validation-cost saved epoch
            cnn.load_best_val_epoch();
            
            % estimated-outputs
            % cnn.net.conserveMemory = false;
            y_ = cnn.out(cnn.db.x);
            
            % save
            save(fullfile(cnn.props.data.bak_dir, 'y_.mat'), 'y_');
        end
        
        function param = analyze_param_name(param_name)
            % Analyze parameter name
            %
            % Parameters
            % ----------
            % - param_name: char vector
            %   Name of parameter
            %
            % Returns
            % -------
            % - param: struct('is_bias', boolean, 'title', char vector)
            %   Parameter info such as `is_bias` and `title`
            titles = struct(...
                'B', 'Bipolar', ...
                'A', 'Amacrine', ...
                'G', 'Ganglion' ...
            );
        
            param.is_bias = (param_name(1) == 'b');
            param.title = titles.(param_name(3));
        end
        
        % todo: move to another `lib` or `methods group`
        function rms_db(db_path)
            % Compute the `root mean square` of given `data-base`
            %
            % Parameters
            % ----------
            % - db_path: char vector
            %   Path of database
            
            % load `db`
            db = load(db_path);
            % rms(`db.y`)
            rms_y = cellfun(@rms, db.y);
            % print `mean` and `std`
            fprintf('rms(db.y) = %d (%d)\n', mean(rms_y), std(rms_y));
        end
    end
    
    % Plot
    methods
        function plot(obj, x, y, plot_color, plot_title, x_label, y_label)
            % Like as matlab `plot`
            %
            % Parameters
            % ----------
            % - x: double vector
            %   `x` input
            % - y: double vector
            %   `y` input
            % - plot_color: char vector
            %   Color of plot
            % - plot_title: char vector
            %   Title of plot
            % - x_label: char vector
            %   Label of `x` axis
            % - y_label: char vector
            %   Label of `y` axis
            
            plot(x, y, 'Color', plot_color);
            obj.title(plot_title);
            xlabel(x_label);
            ylabel(y_label);
            %   - ticks
            %       - x
            obj.twoticks(x, 'XTick');
            %       - y
            obj.twoticks(y, 'YTick');
            %   - grid
            grid('on');
            box('off');
        end
        
        function plot_allinone(obj, ax, x)
            % Plot all data in one figure
            %
            % Parameters
            % ----------
            % - ax: Axes
            %   axes handle
            % - x: cell array
            %   input data {x1, x2, ...}
            
            if nargin == 2
                % replace 'x' with 'ax'
                x = ax;
                
                % figure
                DagNNViz.figure('All in One');
                ax = gca;
            end
            
            % number of samples
            N = length(x);
            
            % plot
            hold('on');
            for i = 1 : N
                plot(ax, x{i});
            end
            obj.title(sprintf('%d Samples', N));
            hold('off');
            
            axis('tight');
        end
        
        function plot_populationmean(obj, ax, x)
            % Plot mean of population
            %
            % Parameters
            % ----------
            % - ax: Axes
            %   axes handle
            % - x: cell array
            %   input data {x1, x2, ...}
            
            if nargin == 2
                % replace 'x' with 'ax'
                x = ax;
                
                % figure
                DagNNViz.figure('Population Mean');
                ax = gca;
            end
            
            plot(ax, DagNNViz.populationmean(x));
            obj.title('Mean');
        end
        
        function plot_summary(obj, x)
            % Plot all data in one figure
            %
            % Parameters
            % ----------
            % - x: cell array
            %   input data  {x1, x2, ...}
            
            % number of samples
            N = length(x);
            
            % figure
            DagNNViz.figure('Summary');
            
            % subplot grid
            % - number of rows
            rows = 4;
            % - number of columns for first part (bigger part)
            cols1 = 3;
            % - number of columns for second part (smaller parts)
            cols2 = 1;
            cols = cols1 + cols2;
            % indexes
            indexes = reshape(1 : (rows * cols), [cols, rows])';
            % - indexes of first part (bigger part)
            indexes1 = indexes(:, 1 : cols1);
            indexes1 = sort(indexes1(:));
            % - indexes of second part (smaller parts)
            indexes2 = indexes(:, (cols1 + 1) : end);
            indexes2 = sort(indexes2(:));

            % plot
            % - first part (bigger part)
            obj.plot_allinone(...
                subplot(rows, cols, indexes1), ...
                x ...
            );
            
            % - second part (smaller parts)
            %   - first sample
            subplot(rows, cols, indexes2(1));
            plot(x{1});
            obj.title(sprintf('Sample #%d', 1));
            
            %   - middle sample
            subplot(rows, cols, indexes2(2));
            %       - index of middle sample
            middle_index = max(floor(N/2), 1);
            plot(x{middle_index});
            obj.title(sprintf('Sample #%d',middle_index));
            
            %   - last sample
            subplot(rows, cols, indexes2(3));
            plot(x{N});

            obj.title(sprintf('Sample #%d', N));
            
            %   - mean of samples
            obj.plot_populationmean(...
                subplot(rows, cols, indexes2(4)), ...
                x ...
            ); 
        end
        
        function boxplot_db(obj, db)
            % Boxplot of db
            
            % figure
            DagNNViz.figure('Box Plot - DB');
            
            % stimulus
            subplot(121);
            boxplot(...
                vertcat(db.x{:}), ...
                'Notch', 'off', ...
                'Labels', {'Stimulus'} ...
            );
            ylabel('Intensity');
            
            % response
            subplot(122);
            boxplot(...
                vertcat(db.y{:}), ...
                'Notch', 'off', ...
                'Labels', {'Response'} ...
            );
            ylabel('Rate (Hz)');
            
            % title
            obj.suptitle('Box Plot of Data (Stimulus/Response)');
        end
        
        function plot_db_all(obj, db, small_db_size)
            % Plot all input/output samples in grid
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   Input database
            % - small_db_size: int (default = 50)
            %   Select first samples from db
            
            DagNNViz.print_title('Plot DB All');
            
            % figure
            DagNNViz.figure('DB All');
        
            % number of samples
            if ~exist('small_db_size', 'var')
                small_db_size = min([50, length(db.x), length(db.y)]);
            end
            N = small_db_size;
            N2 = 2 * N;
            
            % subplot grid
            % - cols
            cols = floor(sqrt(N2));
            % - cols must be even
            if mod(cols, 2) == 1
                cols = cols + 1;
            end
            % - rows
            rows = floor(N2 / cols);
            % - change N, N2
            N2 = rows * cols;
            N = N2 / 2;
            
            % plot
            i = 1;
            % first input/output pair
            fontsize = 7;
            % - sample index
            j = floor((i + 1) / 2);
            % - input
            %   - print progress
            fprintf('Sample %d / %d\n', i, N2);
            subplot(rows, cols, i);
            plot(db.x{j}, 'Color', 'blue');
            obj.title('Stimulus', 'FontSize', fontsize + 2);
            xlabel('Time (s)', 'FontSize', fontsize);
            ylabel('Intensity', 'FontSize', fontsize);
            DagNNViz.hideticks();
            % - output
            %   - print progress
            fprintf('Sample %d / %d\n', i + 1, N2);
            
            subplot(rows, cols, i + 1);
            plot(db.y{j}, 'Color', 'red');
            obj.title('Response', 'FontSize', fontsize + 2);
            xlabel('Time (s)', 'FontSize', fontsize);
            ylabel('Rate (Hz)', 'FontSize', fontsize);
            DagNNViz.hideticks();
            % - other samples
            for i = 3:2:N2
                % - sample index
                j = floor((i + 1) / 2);
                
                % - input
                %   - print progress
                fprintf('Sample %d / %d\n', i, N2);
                
                subplot(rows, cols, i);
                plot(db.x{j}, 'Color', 'blue');
                DagNNViz.hideticks();
            
                % - output
                %   - print progress
                fprintf('Sample %d / %d\n', i + 1, N2);

                subplot(rows, cols, i + 1);
                plot(db.y{j}, 'Color', 'red');
                DagNNViz.hideticks();
            end
            % super-title
            obj.suptitle(...
                sprintf(...
                    'First %d Samples of %d (Stimulous/Response) Pairs of Training Set', ...
                    N, ...
                    length(db.x) ...
                ) ...
            );
            % save
            obj.saveas('db_all');
        end
        
        function plot_bias(obj, bak_dir, param_name, title_txt)
            % Plots history of `param_name` bias based on saved
            % epochs in `bak_dir` directory
            %
            % Parameters
            % ----------
            % - bak_dir: char vector
            %   path of directory of saved epochs
            % - param_name: char vector
            %   name of target parameter
            % - title_txt: char vector
            %   title of plot
            
            % get history of prameter
            param_history = DagNNViz.get_param_history(bak_dir, param_name);
            param_history = [param_history{:}];
            
            % plot
            plot(param_history);
            xlabel('Epoch');
            ylabel('Bias');
            
            obj.title(title_txt);
        end
        
        % todo: make methods('unused') and send methods like this one to it
        function save_video(obj, x, filename, frame_rate)
            % Save data 'x' as a 'filename.mp4' vidoe file
            %
            % Parameters
            % ----------
            % - x: cell array
            %   input data
            % - filename: char vector
            %   name of saved video
            % - frame_rate: int (default is 15)
            %   frame-rate of saved video
            
            % defualt frame-rate is 15
            if ~exist('frame_rate', 'var')
                frame_rate = 15;
            end
            
            % open video writer
            vw = VideoWriter(filename, 'MPEG-4');
            vw.FrameRate = frame_rate;
            open(vw);
            
            % figure
            h = DagNNViz.figure('Video');
        
            % number of samples
            N = length(x);
            
            % delay
            delay = 1 / frame_rate;
            
            % make video
            ylimits = DagNNViz.get_ylimits(x);
            for i = 1 : N
                plot(x{i});
                ylim(ylimits);
                obj.title(sprintf('#%d / #%d', i, N));
                writeVideo(vw, getframe(h));
                
                pause(delay);
            end
            
            % close video writer
            close(vw);
        end
       
        function save_db_video(obj, db, filename, frame_rate)
            % Save database 'db' as a 'filename.mp4' vidoe file
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   input database
            % - filename: char vector
            %   name of saved video
            % - frame_rate: int (default is 15)
            %   frame-rate of saved video
            
            % defualt frame-rate is 15
            if ~exist('frame_rate', 'var')
                frame_rate = 15;
            end
            
            % open video writer
            vw = VideoWriter(filename, 'MPEG-4');
            vw.FrameRate = frame_rate;
            open(vw);
            
            % figure
            h = DagNNViz.figure('Video');
        
            % number of samples
            N = min(length(db.x), length(db.y));
            
            % delay
            delay = 1 / frame_rate;
            
            % make video
            for i = 1 : N
                % - input
                subplot(121), plot(db.x{i}, 'Color', 'blue');
                obj.title(sprintf('Input (#%d / #%d)', i, N));
                
                % - output
                subplot(122), plot(db.y{i}, 'Color', 'red');
                obj.title(sprintf('Output (#%d / #%d)', i, N));
                
                % - frame
                writeVideo(vw, getframe(h));
                
                % - delay
                pause(delay);
            end
            
            % close video writer
            close(vw);
        end
        
        function save_db_yhat_video(obj, db, y_, filename, frame_rate)
            % Save 'db.x', 'db.y', 'y^' as a
            % 'filename.mp4' video file
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   input database
            % - y_: cell array
            %   estimated outputs
            % - filename: char vector
            %   name of saved video
            % - frame_rate: int (default is 15)
            %   frame-rate of saved video
            
            % defualt frame-rate is 15
            if ~exist('frame_rate', 'var')
                frame_rate = 15;
            end
            
            % open video writer
            vw = VideoWriter(filename, 'MPEG-4');
            vw.FrameRate = frame_rate;
            open(vw);
            
            % figure
            h = DagNNViz.figure('Video');
        
            % number of samples
            N = min(length(db.x), length(db.y));
            
            % delay
            delay = 1 / frame_rate;
            
            % make video
            for i = 1 : N
                % - input
                subplot(131), plot(db.x{i}, 'Color', 'blue');
                obj.title(sprintf('Input (#%d / #%d)', i, N));
                
                % - expected-output
                subplot(132), plot(db.y{i}, 'Color', 'red');
                obj.title(sprintf('Expected-Output (#%d / #%d)', i, N));
                
                % - expected-output
                subplot(133), plot(y_{i}, 'Color', 'green');
                obj.title(sprintf('Estimated-Output (#%d / #%d)', i, N));
                
                % - frame
                writeVideo(vw, getframe(h));
                
                % - delay
                pause(delay);
            end
            
            % close video writer
            close(vw);
        end
        
        function save_frames(obj, x, frames_dir)
            % Save data 'x' as a 'sample#.png' image files
            %
            % Parameters
            % ----------
            % - x: cell array
            %   Input data
            % - frames_dir: char vector
            %   Path of output directory for saving frames
            
            % directory for save frames
            if exist(frames_dir, 'dir')
                rmdir(frames_dir, 's');
            end
            mkdir(frames_dir);
            
            % figure
            h = DagNNViz.figure('Video');
        
            % number of samples
            N = length(x);
            
            % delay
            delay = 0.01;
            
            % save images
            for i = 1 : N
                plot(x{i});
                obj.title(sprintf('#%d / #%d', i, N));
            
                saveas(...
                    h, ...
                    fullfile(frames_dir, [num2str(i), '.png']), ...
                    'png' ...
                );
                
                pause(delay);
            end
        end
        
        function save_db_frames(obj, db, frames_dir)
            % Save database 'db' as a 'sample#.png' image files
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   Input database
            % - frames_dir: char vector
            %   Path of output directory for saving frames
            
            % directory for save frames
            if exist(frames_dir, 'dir')
                rmdir(frames_dir, 's');
            end
            mkdir(frames_dir);
            
            % figure
            h = DagNNViz.figure('Video');
        
            % number of samples
            N = min(length(db.x), length(db.y));
            
            % delay
            delay = 0.01;
            
            % save images
            for i = 1 : N
                % - input
                subplot(121), plot(db.x{i}, 'Color', 'blue');
                obj.title(sprintf('Input (#%d / #%d)', i, N));
                
                % - output
                subplot(122), plot(db.y{i}, 'Color', 'red');
                obj.title(sprintf('Output (#%d / #%d)', i, N));
                
                % - save image to file
                saveas(...
                    h, ...
                    fullfile(frames_dir, [num2str(i), '.png']), ...
                    'png' ...
                );
                
                % - pause
                pause(delay);
            end
        end
        
        function plot_spks(obj, experimets_dir)
            % Plot 'spk' data from saved 'experiment' files in
            % 'experiments_dir' directory
            %
            % Parameters
            % ----------
            % - experiments_dir: char vector
            %   Path of saved 'experiments' files
            
            % spks
            spks = DagNNViz.get_spks(experimets_dir);
            % number of elements
            N = length(spks);
            % add number_of_spiks field
            for i = 1 : N
                spks(i).number_of_spiks = sum(spks(i).value);
            end
            % convert to table
            T = struct2table(spks);
            % sort table based on 'number_of_spiks' columns
            T = sortrows(T, 'number_of_spiks', 'descend');
            
            % subplot grid
            % - rows
            rows = ceil(sqrt(N));
            % - cols
            cols = rows;
            
            % plot
            for i = 1 : N
                subplot(rows, cols, i);
                DagNNViz.plot_spike_trains(T{i, 'value'});
                
                % - title (number of spikes)
                obj.title(...
                    sprintf(...
                        '%d Spikes\n%s', ...
                        T{i, 'number_of_spiks'} ...
                    ) ...
                );
                
                xlabel('');
                ylabel(...
                    char(...
                        regexp(...
                            char(T{i, 'name'}), ...
                            'c\d+', ...
                            'match' ...
                        ) ...
                    ) ...
                );
                
                set(gca, ...
                    'XTick', [], ...
                    'YTick', [] ...
                );
            end
            
        end
        
        function plot_params(obj, param_names, bak_dir, number_of_epochs, dt_sec)
            % Plot and save parameters in 'params.mat' file
            %
            % Parameters
            % ----------
            % - bak_dir: char vector
            %   Path of directory of saved epochs
            % - number_of_epochs: int
            %   Number of epochs
            % - dt_sect: double
            %   Time resolution in second
            
            % - prameter names
            % param_names = {'w_B', 'w_A', 'w_G', 'b_B', 'b_A', 'b_G'};
            
            % costs
            % - load
            costs = load(fullfile(bak_dir, 'costs'));
            % - bets validation index
            [~, index_min_val_cost] = min(costs.val);
            
            % plot and save
            for i = 1 : length(param_names)
                % param
                % param.isbias, param.title
                param = DagNNViz.analyze_param_name(param_names{i});
                % param.value
                param.value = ...
                    DagNNViz.get_param_history(bak_dir, (param_names{i}));
                
                if isempty(param.value)
                    continue
                end
                
                % - resize param.values
                param.value = param.value(1 : number_of_epochs);
                
                % new figure
                DagNNViz.figure('Parameters');
                
                % if parameter is a bias
                if param.is_bias
                    param.value = [param.value{:}];
                    epochs = 0 : (length(param.value) - 1);
                    plot(epochs, param.value);
                    hold('on');
                    plot(...
                        index_min_val_cost - 1, ...
                        round(param.value(index_min_val_cost), obj.round_digits), ...
                        'Color', 'red', ...
                        'Marker', '*', ...
                        'MarkerSize', 3, ...
                        'LineWidth', 2 ...
                    );
                    grid('on');
                    box('off');
                    obj.title(sprintf('%s (Bias with Min Validation Cost is Red)', param.title));
                    xlabel('Epoch');
                    ylabel('Bias');
                    
                    % ticks
                    % - x
                    %todo: writ `threeticks` method
                    set(gca, ...
                        'XTick', ...
                        unique([...
                            0, ...
                            index_min_val_cost - 1, ...
                            number_of_epochs ...
                        ]) ...
                    );
                    % - y
                    set(gca, ...
                        'YTick', ...
                        unique([...
                            round(min(param.value), obj.round_digits), ...
                            round(param.value(index_min_val_cost), obj.round_digits), ...
                            round(max(param.value), obj.round_digits) ...
                        ]) ...
                    );
                    ylim([round(min(param.value), obj.round_digits), round(max(param.value), obj.round_digits)]);
                    
                    % save
                    obj.saveas(['bias_', lower(param.title)]);
                
                % if parameter is a filter
                else
                    % - all
                    obj.plot_filter_history(param.value, index_min_val_cost);
                    box('off');
                    obj.suptitle(...
                        sprintf(...
                            'Kernel of %s for each Epoch of Training (Kernel with Min Validation Cost is Red)', ...
                            param.title ...
                        ) ...
                    );
                
                    % save
                    obj.saveas(['filter_', lower(param.title), '_all']);
                
                    % - initial & best
                    obj.plot_filter_initial_best(param.value, index_min_val_cost, dt_sec);
                    box('off');
                    title(sprintf('Kernel of %s', param.title));
                    
                    % save
                    obj.saveas(['filter_', lower(param.title), '_initial_best']);
                end
            end
        end
        
        function plot_stim(obj, stim, dt_sec)
            % Plot stimulous
            %
            % Parameters
            % ----------
            % - stim: double vector
            %   Stimulus
            % - dt_sect: double
            %   Time resolution in second
            
            time = (1 : length(stim)) * dt_sec;
            
            % figure
            DagNNViz.figure('STIM');
            % plot
            plot(time, stim);
            % - title
            obj.title('Stimulus');
            % - label
            %   - x
            xlabel('Time (s)');
            %   - y
            ylabel('Intensity');
            % - ticks
            %   - x
            set(gca, ...
                'XTick', [0, round(time(end), 1)] ...
            );
            %   - y
            set(gca, ...
                'YTick', unique([round(min(stim), 1), 0, round(max(stim), 1)]) ...
            );
            % - grid
            grid('on');
            
            % save
            obj.saveas('stim');
        end
        
        function plot_costs(obj, costs)
            % Plot 'costs' over time
            %
            % Parameters
            % ----------
            % - costs: ?
            %   
            
            epochs = 1:length(costs.train);
            % start epochs from zero (0, 1, 2, ...)
            epochs = epochs - 1;
            
            DagNNViz.figure('CNN - Costs [Training, Validation, Test]');
            
            % costs
            % - train
            plot(epochs, costs.train, 'LineWidth', 2, 'Color', 'blue');
            set(gca, 'YScale', 'log');
            hold('on');
            % - validation
            plot(epochs, costs.val, 'LineWidth', 2, 'Color', 'green');
            % - test
            plot(epochs, costs.test, 'LineWidth', 2, 'Color', 'red');
            
            % minimum validation error
            % - circle
            [~, index_min_val_cost] = min(costs.val);
            circle_x = index_min_val_cost - 1;
            circle_y = costs.val(index_min_val_cost);
            dark_green = [0.1, 0.8, 0.1];
            scatter(circle_x, circle_y, ...
                'MarkerEdgeColor', dark_green, ...
                'SizeData', 75, ...
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
            
            % ticks
            % - x
            set(gca, ...
                'XTick', unique([0, index_min_val_cost - 1, epochs(end)]) ...
            );
            % - y
            set(gca, ...
                'YTick', ...
                unique([...
%                     round(min([costs.train, costs.val, costs.test]), obj.round_digits), ...
                    round(costs.val(index_min_val_cost), obj.round_digits), ...
                    round(max([costs.train, costs.val, costs.test]), obj.round_digits) ...
                ]) ...
            );
            
            % labels
            xlabel('Epoch');
            ylabel('Mean Squared Error (Hz^2)');
            
            % title
            obj.title(...
                sprintf('Minimum Validation Error is %g at Epoch: %d', ...
                costs.val(index_min_val_cost), ...
                index_min_val_cost - 1 ...
                ) ...
            );
            
            % legend
            legend(...
                sprintf('Training (%g)', costs.train(index_min_val_cost)), ...
                sprintf('Validation (%g)', costs.val(index_min_val_cost)), ...
                sprintf('Test (%g)', costs.test(index_min_val_cost)), ...
                'Best Validation Error' ...
            );
            
            % grid
            grid('on');
            
            % box
            box('off');
            
            % save
            obj.saveas('error');
        end
        
        function plot_dcosts(obj, costs)
            % Plot derivative of 'costs' over time
            %
            % Parameters
            % ----------
            % - costs: ?
            %   
            
            epochs = 1:length(costs.train) - 1;
            
            DagNNViz.figure('CNN - Derivatif of Costs [Training, Validation, Test]');
            
            % costs
            % - train
            plot(epochs, diff(costs.train), 'LineWidth', 2, 'Color', 'blue');
            set(gca, 'YScale', 'log');
            hold('on');
            % - validation
            plot(epochs, diff(costs.val), 'LineWidth', 2, 'Color', 'green');
            % - test
            plot(epochs, diff(costs.test), 'LineWidth', 2, 'Color', 'red');
            
            % minimum validation error
            % - circle
%             [~, index_min_val_cost] = min(costs.val);
%             circle_x = index_min_val_cost - 1;
%             circle_y = costs.val(index_min_val_cost);
%             dark_green = [0.1, 0.8, 0.1];
%             scatter(circle_x, circle_y, ...
%                 'MarkerEdgeColor', dark_green, ...
%                 'SizeData', 75, ...
%                 'LineWidth', 2 ...
%             );
%             
%             % - cross lines
%             h_ax = gca;
%             %   - horizontal line
%             line(...
%                 h_ax.XLim, ...
%                 [circle_y, circle_y], ...
%                 'Color', dark_green, ...
%                 'LineStyle', ':', ...
%                 'LineWidth', 1.5 ...
%             );
%             %   - vertical line
%             line(...
%                 [circle_x, circle_x], ...
%                 h_ax.YLim, ...
%                 'Color', dark_green, ...
%                 'LineStyle', ':', ...
%                 'LineWidth', 1.5 ...
%             );
%             
%             hold('off');
            
            % ticks
%             % - x
%             set(gca, ...
%                 'XTick', unique([0, index_min_val_cost - 1, epochs(end)]) ...
%             );
%             % - y
%             set(gca, ...
%                 'YTick', ...
%                 unique([...
% %                     round(min([costs.train, costs.val, costs.test]), obj.round_digits), ...
%                     round(costs.val(index_min_val_cost), obj.round_digits), ...
%                     round(max([costs.train, costs.val, costs.test]), obj.round_digits) ...
%                 ]) ...
%             );
            
            % labels
            xlabel('Epoch');
            ylabel('Mean Squared Error (Hz^2)');
            
            % title
            obj.title(...
                sprintf('Minimum Validation Error is %g at Epoch: %d', ...
                costs.val(index_min_val_cost), ...
                index_min_val_cost - 1 ...
                ) ...
            );
            
            % legend
            legend(...
                sprintf('Training (%g)', costs.train(index_min_val_cost)), ...
                sprintf('Validation (%g)', costs.val(index_min_val_cost)), ...
                sprintf('Test (%g)', costs.test(index_min_val_cost)), ...
                'Best Validation Error' ...
            );
            
            % grid
            grid('on');
            
            % box
            box('off');
        end
        
        function plot_ddcosts(obj, costs)
            % Plot derivative of 'costs' over time
            %
            % Parameters
            % ----------
            % - costs: ?
            %   
            
            epochs = 2:length(costs.train) - 1;
            
            DagNNViz.figure('CNN - Derivatif of Costs [Training, Validation, Test]');
            
            % costs
            % - train
            plot(epochs, diff(diff(costs.train)), 'LineWidth', 2, 'Color', 'blue');
            set(gca, 'YScale', 'log');
            hold('on');
            % - validation
            plot(epochs, diff(diff(costs.val)), 'LineWidth', 2, 'Color', 'green');
            % - test
            plot(epochs, diff(diff(costs.test)), 'LineWidth', 2, 'Color', 'red');
            
            % minimum validation error
            % - circle
%             [~, index_min_val_cost] = min(costs.val);
%             circle_x = index_min_val_cost - 1;
%             circle_y = costs.val(index_min_val_cost);
%             dark_green = [0.1, 0.8, 0.1];
%             scatter(circle_x, circle_y, ...
%                 'MarkerEdgeColor', dark_green, ...
%                 'SizeData', 75, ...
%                 'LineWidth', 2 ...
%             );
%             
%             % - cross lines
%             h_ax = gca;
%             %   - horizontal line
%             line(...
%                 h_ax.XLim, ...
%                 [circle_y, circle_y], ...
%                 'Color', dark_green, ...
%                 'LineStyle', ':', ...
%                 'LineWidth', 1.5 ...
%             );
%             %   - vertical line
%             line(...
%                 [circle_x, circle_x], ...
%                 h_ax.YLim, ...
%                 'Color', dark_green, ...
%                 'LineStyle', ':', ...
%                 'LineWidth', 1.5 ...
%             );
%             
%             hold('off');
            
            % ticks
%             % - x
%             set(gca, ...
%                 'XTick', unique([0, index_min_val_cost - 1, epochs(end)]) ...
%             );
%             % - y
%             set(gca, ...
%                 'YTick', ...
%                 unique([...
% %                     round(min([costs.train, costs.val, costs.test]), obj.round_digits), ...
%                     round(costs.val(index_min_val_cost), obj.round_digits), ...
%                     round(max([costs.train, costs.val, costs.test]), obj.round_digits) ...
%                 ]) ...
%             );
            
            % labels
            xlabel('Epoch');
            ylabel('Mean Squared Error (Hz^2)');
            
            % title
            obj.title(...
                sprintf('Minimum Validation Error is %g at Epoch: %d', ...
                costs.val(index_min_val_cost), ...
                index_min_val_cost - 1 ...
                ) ...
            );
            
            % legend
            legend(...
                sprintf('Training (%g)', costs.train(index_min_val_cost)), ...
                sprintf('Validation (%g)', costs.val(index_min_val_cost)), ...
                sprintf('Test (%g)', costs.test(index_min_val_cost)), ...
                'Best Validation Error' ...
            );
            
            % grid
            grid('on');
            
            % box
            box('off');
        end
        
        function plot_all(obj, x, red_index)
            % Plot all samples in square grid
            %
            % Parameters
            % ----------
            % - x: cell array
            %   input data {x1, x2, ...}
            % - red_index: int (default = 0)
            %   index of 'red' sample
            
            DagNNViz.print_title('Plot All');
            
            % default values
            if nargin < 2
                red_index = 0;
            end
            
            % figure
            DagNNViz.figure('All');
            
            % number of samples
            N = length(x);
            
            % subplot grid
            % - rows
            rows = ceil(sqrt(N));
            % - cols
            cols = rows;
            
            % plot
            fontsize = 7;
            % - first sample
            i = 1;
            % print progress
            fprintf('Sample %d / %d\n', i, N);

            subplot(rows, cols, i);
            h = plot(x{i});
            DagNNViz.hideticks();
            xlabel('Time (s)', 'FontSize', fontsize);

            % red sample
            if i == red_index
               set(h, 'Color', 'red'); 
            end
            
            % - other samples
            for i = 2 : N
                % print progress
                fprintf('Sample %d / %d\n', i, N);
                
                subplot(rows, cols, i);
                h = plot(x{i});
                DagNNViz.hideticks();
                
                % red sample
                if i == red_index
                   set(h, 'Color', 'red'); 
                end
            end

            obj.suptitle(sprintf('%d Samples', N));
        end
        
        % todo: refactor to `plot_all`
        function plot_filter_history(obj, x, red_index)
            % Plot all samples in square grid
            %
            % Parameters
            % ----------
            % - x: cell array
            %   input data
            % - red_index: int (default = 0)
            %   index of 'red' sample
            
            DagNNViz.print_title('Plot Filter History');
            
            % default values
            if nargin < 2
                red_index = 0;
            end
            
            % number of samples
            N = length(x);
            
            % subplot grid
            % - rows
            rows = ceil(sqrt(N));
            % - cols
            cols = rows;
            
            % plot
            fontsize = 7;
            % - first sample
            i = 1;
            % print progress
            fprintf('Sample %d / %d\n', i, N);

            subplot(rows, cols, i);
            h = plot(x{i});
            DagNNViz.hideticks();
            title('Initial Value', 'FontSize', fontsize + 2);
            xlabel('Time (s)', 'FontSize', fontsize);

            % red sample
            if i == red_index
               set(h, 'Color', 'red'); 
            end
            
            % - other samples
            for i = 2 : N
                % print progress
                fprintf('Sample %d / %d\n', i, N);
                
                subplot(rows, cols, i);
                h = plot(x{i});
                DagNNViz.hideticks();
                
                % red sample
                if i == red_index
                   set(h, 'Color', 'red'); 
                end
            end
            obj.suptitle(sprintf('%d Samples', N));
        end
        
        function plot_db_first(obj, db, dt_sec)
            % Plot first input/output sample
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   input database
            % - dt_sect: double
            %   time resolution in second
            
            % figure
            DagNNViz.figure('DB - First Sample Pair');
        
            % subplot grid
            % - rows
            rows = 1;
            % - cols
            cols = 2;
            
            % first input/output pair
            % - input
            subplot(rows, cols, 1);
            time = (0 : length(db.x{1}) - 1) * dt_sec;
            obj.plot(...
                time, ...
                db.x{1}, ...
                'blue', ...
                'Stimulus', ...
                'Time (s)', ...
                'Intensity' ...
            );
            % - output
            subplot(rows, cols, 2);
            time = (0 : length(db.y{1}) - 1) * dt_sec;
            obj.plot(...
                time, ...
                db.y{1}, ...
                'red', ...
                'Response', ...
                'Time (s)', ...
                'Rate (Hz)' ...
            );
            % super-title
            obj.suptitle(...
                sprintf('First Sample (Stimulous/Response) of Training Set') ...
            );
            % save
            obj.saveas('db_first');
        end
        
        % todo: use `obj.dt_sec` instead of input parameter
        function plot_db_sample(obj, db, dt_sec, sample_number)
            % Plot first input/output sample
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   input database
            % - dt_sect: double
            %   time resolution in second
            % - sample_numer: int = 1
            %   number of target sample
            
            % default values
            if (~exist('sample_number', 'var'))
                sample_number = 1;
            end
            
            % figure
            DagNNViz.figure(sprintf('DB - Sample #%d Pair', sample_number));
        
            % subplot grid
            % - rows
            rows = 1;
            % - cols
            cols = 2;
            
            % first input/output pair
            % - input
            subplot(rows, cols, 1);
            time = (0 : length(db.x{1}) - 1) * dt_sec;
            obj.plot(...
                time, ...
                db.x{sample_number}, ...
                'blue', ...
                'Stimulus', ...
                'Time (s)', ...
                'Intensity' ...
            );
            % - output
            subplot(rows, cols, 2);
            time = (0 : length(db.y{1}) - 1) * dt_sec;
            obj.plot(...
                time, ...
                db.y{sample_number}, ...
                'red', ...
                'Response', ...
                'Time (s)', ...
                'Rate (Hz)' ...
            );
            % super-title
            obj.suptitle(...
                sprintf(...
                    'Sample #%d (Stimulous/Response) of Training Set', ...
                    sample_number ...
                ) ...
            );        
        end
        
        function plot_db_yhat_all(obj, db, y_, small_db_size)
            % Plot all 'db.x', 'db.y' and 'y^' samples in
            % square grid
            %
            % Parameters
            % ----------
            % - db: struct('x', cell array, 'y', cell array)
            %   input database
            % - y_: cell array
            %   estimated outputs
            % - small_db_size: int (default = 50)
            %   Select first samples from db
            
            DagNNViz.print_title('Plot DB and Actual `y` All');
            
            % number of samples
            if ~exist('small_db_size', 'var')
                small_db_size = min([27, length(db.x), length(db.y), length(y_)]);
            end
            N = small_db_size;
            N3 = 3 * N;
            
            % subplot grid
            % - rows
            rows = ceil(sqrt(N3));
            % - cols
            cols = rows;
            % - cols (mod(cols, 3) must be 0)
            if mod(cols, 3) == 1
                cols = cols + 2;
            elseif mod(cols, 3) == 2
                cols = cols + 1;
            end
            
            % plot
            for i = 1:3:N3
                % - sample index
                j = floor((i + 2) / 3);
                % - input
                subplot(rows, cols, i);
                plot(db.x{j}, 'Color', 'blue');
                DagNNViz.hideticks();
            
                % - expected-output
                subplot(rows, cols, i + 1);
                plot(db.y{j}, 'Color', 'red');
                DagNNViz.hideticks();
            
                % - estimated-output
                subplot(rows, cols, i + 2);
                plot(y_{j}, 'Color', 'green');
                DagNNViz.hideticks();
            end
            obj.suptitle(sprintf('%d Samples', N));
        end
        
        function plot_resp(obj, resp, dt_sec)
            % Plot response
            %
            % Parameters
            % ----------
            % - stim: double vector
            %   Stimulus
            % - dt_sect: double
            %   Time resolution in second
            
            time = (1 : length(resp)) * dt_sec;
            
            % figure
            DagNNViz.figure('RESP');
            % plot
            plot(time, resp);
            % - title
            obj.title('PSTH');
            % - label
            %   - x
            xlabel('Time (s)');
            %   - y
            ylabel('Firing Rate (Hz)');
            % - ticks
            %   - x
            set(gca, ...
                'XTick', [0, round(time(end), 1)] ...
            );
            %   - y
            set(gca, ...
                'YTick', unique([round(min(resp), 1), round(max(resp), 1)]) ...
            );
            % - grid
            grid('on');
            
            % save
            obj.saveas('resp');
        end
        
        function plot_noisy_params(obj, props_filename, params_path, noisy_params_path, snr, dt_sec)
            % Plot noisy params with target ones
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of `properties`
            % - params_path: char vector
            %   Path of noiseless `params` file
            % - noisy_params_path: char vector
            %   Path of noisy `params` file
            % - snr: double
            %   Signal-to-noise ratio per sample, in dB
            % - dt_sec: double
            %   time resolution in seconds
            
            if ~exist('dt_sec', 'var')
                dt_sec = Neda.dt_sec;
            end
            
            % props
            props = jsondecode(fileread(props_filename));
            
            % params
            % - noisless
            noiselessParams = load(params_path);
            % - noisy
            noisyParams = load(noisy_params_path);

            for i = 1 : length(props.net.params)
                name = props.net.params(i).name;
                size = props.net.params(i).size'; 
                paramName = DagNNViz.analyze_param_name(name);
                
                % todo: write a `isBias` function
                if paramName.is_bias
                    continue
                end
                
                % figure
                DagNNViz.figure('Noisy Parameters');
            
                % noisless/noisy filter
                % - noisless
                noiseless = DataUtils.resize(...
                    noiselessParams.(name), ...
                    size ...
                );
                time = (0 : length(noiseless) - 1) * dt_sec;
                plot(time, noiseless, 'Color', 'blue');
                
                hold('on');
                
                % - noisy
                noisy = DataUtils.resize(...
                    noisyParams.(name), ...
                    size ...
                );
                plot(time, noisy, 'Color', 'red');
                
                title(sprintf('Kernel of %s', paramName.title));
                xlabel('Time (s)');
                ylabel('');
                %   - ticks
                %       - x
                set(gca, ...
                    'XTick', unique([0, round(time(end), obj.round_digits)]) ...
                );
                %       - y
%                 set(gca, ...
%                     'YTick', ...
%                     unique([...
%                         round(min(min(noiseless), min(noisy)), obj.round_digits), ...
%                         round(max(max(noiseless), max(noisy)), obj.round_digits) ...
%                     ]) ...
%                 );
                % todo: write `round` function
                set(gca, ...
                    'YTick', ...
                    unique(sort([...
                        round(min(noiseless), obj.round_digits), ...
                        round(max(noiseless), obj.round_digits), ...
                        round(min(noisy), obj.round_digits), ...
                        round(max(noisy), obj.round_digits) ...
                    ])) ...
                );
                %   - grid
                grid('on');
                box('off');
                
                 % legend
                legend(...
                    'Original', ...
                    sprintf('with AWGN (SNR %0.2f dB)', snr) ...
                );

                % save
                obj.saveas([lower(paramName.title), '_filter_noisy_noiseless']);
            end            
        end
        
        function plot_filter_initial_best_old(obj, x, red_index, dt_sec)
            % Plot `initial` and `best` samples
            %
            % Parameters
            % ----------
            % - x: cell array
            %   input data. x{1} is `initial` and x{red_index} is `best`
            % - red_index: int
            %   index of 'red' sample (`best` one)
            % - dt_sect: double
            %   time resolution in second
            
            % figure
            DagNNViz.figure('Filter - Initial & Best');
        
            % subplot grid
            % - rows
            rows = 1;
            % - cols
            cols = 2;
            
            % initial/best filter
            % - initial
            initial = x{1};
            subplot(rows, cols, 1);
            time = (0 : length(initial) - 1) * dt_sec;
            obj.plot(...
                time, ...
                initial, ...
                'blue', ...
                'Initial Value (Epoch #0)', ...
                'Time (s)', ...
                '' ...
            );
            % todo: use `axis('tight')`
            % axis('tight');
            
            % - best
            best = x{red_index};
            subplot(rows, cols, 2);
            time = (0 : length(best) - 1) * dt_sec;
            obj.plot(...
                time, ...
                best, ...
                'red', ...
                sprintf('Min Validation Cost (Epoch #%d)', red_index - 1), ...
                'Time (s)', ...
                '' ...
            );
        end
        
        function plot_filter_initial_best(obj, x, red_index, dt_sec)
            % Plot `initial` and `best` samples
            %
            % Parameters
            % ----------
            % - x: cell array
            %   input data. x{1} is `initial` and x{red_index} is `best`
            % - red_index: int
            %   index of 'red' sample (`best` one)
            % - dt_sect: double
            %   time resolution in second
            
            % figure
            DagNNViz.figure('Filter - Initial & Best');
            
            % initial/best filter
            % - initial
            initial = x{1};
            time = (0 : length(initial) - 1) * dt_sec;
            obj.plot(...
                time, ...
                initial, ...
                'red', ...
                '', ...
                'Time (s)', ...
                '' ...
            );
            % todo: use `axis('tight')`
            % axis('tight');
            
            % - best
            best = x{red_index};
            time = (0 : length(best) - 1) * dt_sec;
            hold('on');
            obj.plot(...
                time, ...
                best, ...
                'blue', ...
                '', ...
                'Time (s)', ...
                '' ...
            ); 
            hold('off');
            
            % Y-Ticks
            set(gca, ...
                'YTick', ...
                unique(sort([...
                    round(min(initial), obj.round_digits), ...
                    round(max(initial), obj.round_digits), ...
                    round(min(best), obj.round_digits), ...
                    round(max(best), obj.round_digits) ...
                ])) ...
            );
            
            % legend
            legend(...
                'Initial Value (Epoch #0)', ...
                sprintf('Min Validation Cost (Epoch #%d)', red_index - 1) ...
            );
        end
        
        % todo: correct documetation
        function plot_digraph(obj, filename)
            % Plot a directed-graph based on given `json` file
            %
            % Parameters
            % ----------
            % - filename : char vector
            %   Filename of input `json` file
            %
            % Examples
            % --------
            % 1.
            %   >>> filename = './dagnn.json';
            %   >>> dg = DagNNTrainer.plot_digraph(dagnn_filename);
            %   ...
            
            % read 'json' file
            props = jsondecode(fileread(filename));
            
            % make digraph
            dg = DagNNViz.make_digraph(filename);
            
            % figure
            figure(...
                'Name', 'Net', ...
                'NumberTitle', 'off', ...
                'Units', 'normalized', ...
                'OuterPosition', [0, 0, 1, 1] ...
            );
            
        
            % node labels
            labels = {};
            %   name(type) -> type
            expression = '\w+\((?<type>\w+)\)';
            for name = dg.Nodes.Name'
                token_names = regexp(char(name), expression, 'names');
                if isempty(token_names)
                    labels{end + 1} = char(name);
                else
                    labels{end + 1} = token_names.type;
                end
            end
            
            % plot graph
            h = plot(dg, 'NodeLabel', labels);
            title('Structure of Network');
            
            % layout
            layout(h, 'layered', ...
                'Direction', 'right', ...
                'Sources', props.net.vars.input.name, ...
                'Sinks', props.net.vars.cost.name, ...
                'AssignLayers', 'asap' ...
            );
            
            % highlight
            % - input, output
            highlight(h, ...
                {...
                    props.net.vars.input.name, ...
                    props.net.vars.expected_output.name ...
                }, ...
                'NodeColor', 'red' ...
            );
            % - params
            params = {};
            for i = 1 : length(props.net.params)
                params{end + 1} = props.net.params(i).name;
            end
            highlight(h, ...
                params, ...
                'NodeColor', 'green' ...
            );
            % - blocks
            ms = h.MarkerSize;
            blocks = {};
            for i = 1 : length(props.net.layers)
                blocks{end + 1} = ...
                    sprintf(...
                        '%s(%s)', ...
                        props.net.layers(i).name, ...
                        props.net.layers(i).type ...
                    );
            end
            highlight(h, ...
                blocks, ...
                'Marker', 's', ...
                'MarkerSize', 5 * ms ...
            );
            % hide axes
            set(h.Parent, ...
                'XTick', [], ...
                'YTick', [] ...
            );
        
            % save
            obj.saveas('net');
        end
    end
    methods (Static)
        function print_dashline(length)
            % Print specified length dash-line
            %
            % Parameters
            % ----------
            % - length: int
            %   Length of printed dash-line
            %
            % Examples
            % --------
            % 1.
            %   >>> DagNNViz.print_dashline(5)
            %   -----
            
            if ~exist('length', 'var')
                length = 32;
            end
            
            fprintf('%s\n', repmat('-', 1, length));
        end
        
        function print_title(text)
            % Print `text` in command window
            %
            % Parameters
            % ----------
            % - text: char vector
            %   Text of title
            
            fprintf('%s\n', text);
            DagNNViz.print_dashline(numel(text));
        end
        
        function plot_spike_trains(spike_trains, number_of_time_ticks, time_limits)
            %Plot spike train
            %   Parameters
            %   ----------
            %   - spike_trains : double array
            %   - number_of_time_ticks: int (default = 2)
            %   - time_limits : [double, double] (default = [1, length of each trial]) 
            %       [min_time, max_time]

            % default parameters
            if ~exist('number_of_time_ticks', 'var')
                number_of_time_ticks = 2;
            end
            if ~exist('time_limits', 'var')
                time_limits = [1, size(spike_trains, 2)];
            end
            
            hold('on');
            
            % first baseline
            baseline = 0;
            % number of trails
            N = size(spike_trains, 1);
            % length of spike train
            T = size(spike_trains, 2);
            % time axis
            time = linspace(time_limits(1), time_limits(2), T);
            
            % plot spike train
            for trial_index = 1 : N
                for time_index = 1 : T
                    if spike_trains(trial_index, time_index) > 0
                        plot(...
                            [time(time_index), time(time_index)], ...
                            [baseline, baseline + 1], ...
                            'Color', 'blue' ...
                        );
                    end
                end
                
                baseline = baseline + 1.5;
            end
            
            hold('off');
            % set trial-axis limits
            ylim([-0.5, baseline]);
            % set time-axis label
            xlabel('Time(s)');
            % set trial-axis label
            ylabel('Trail');
            % remove trial-axis ticks
            set(gca, ...
                'XTick', linspace(time_limits(1), time_limits(2), number_of_time_ticks), ...
                'YTick', [] ...
            );
        end
        
        function spks = get_spks(experimets_dir)
            % Get 'spk' data from saved 'experiment' files in
            % 'experiments_dir' directory
            %
            % Parameters
            % ----------
            % - experiments_dir: char vector
            %   Path of saved 'experiments' files
            %
            % Returns
            % -------
            % - spks: struct('value', double array, 'name', char vector)
            % array
            %   contains 'spk' data
            
            % experiment files
            ep_files = dir(fullfile(experimets_dir, '*.mat'));
            ep_files = {ep_files.name};
            
            % number of experiments
            N = length(ep_files);
            
            % spks
            spks(N) = struct('value', [], 'name', []);
            for i = 1 : N
                spks(i).value = getfield(...
                    load(fullfile(experimets_dir, ep_files{i})), ...
                    'spk' ...
                );
                spks(i).name = ep_files{i};
            end
        end
        
         % todo: save diagraph
        function dg = make_digraph(filename)
            % Make a directed-graph based on given `json` file
            %
            % Parameters
            % ----------
            % - filename : char vector
            %   Filename of input `json` file
            %
            % Returns
            % - dg : digraph
            %   Directed graph
            %
            % Examples
            % --------
            % 1.
            %   >>> filename = './dagnn.json';
            %   >>> dg = DagNNTrainer.make_digraph(dagnn_filename);
            %   >>> dg.Edges
            %    EndNodes
            %   __________
            %      ...
            
            % read 'json' file
            props = jsondecode(fileread(filename));
            
            % add layers to digraph
            dg = digraph();
            names = {};
            
            for layer = props.net.layers'
                block = sprintf('%s(%s)', layer.name, layer.type);
                
                % add edges
                % - inputs, block
                for x = layer.inputs'
                    dg = addedge(dg, x, block);
                end
                % - params, block
                if ~isempty(layer.params)
                    if ~isempty(strfind(layer.type, '+'))
                        % parms = {{'p1', 'p2'}, []} -> params = {'p1', 'p2'}
                        layer.params = [layer.params{:}];
                    end
                end
                for w = layer.params'
                    dg = addedge(dg, w, block);
                end
                % - block, outputs
                for y = layer.outputs'
                    dg = addedge(dg, block, y);
                end
            end
        end
        
        function plot_data()
            % Plot data
            
            % parameters
            % - viz
            viz = DagNNViz();
            % - time resolution (sec)
            dt_sec = Neda.dt_sec;

            % output directory
            % - path
            viz.output_dir = fullfile(viz.data_dir, 'summary/images');
            % - make if doesn't exist
            if ~exist(viz.output_dir, 'dir')
                mkdir(viz.output_dir);
            end
            
            % data
            data = load(fullfile(viz.data_dir, 'data.mat'));
            % - stim
            stim = data.stim;
            obj.plot_stim(stim, dt_sec);
            % - resp
            resp = data.resp;
            obj.plot_resp(resp, dt_sec);
            
            % db
            % - all
            db = load(fullfile(viz.data_dir, 'db.mat'));
            small_db_size = 50;
            viz.plot_db_all(...
                db, ...
                small_db_size ...
            );
            % - first
            viz.plot_db_first(...
                db, ...
                dt_sec ...
            );
            
        end
        
        function plot_results(props_filename)
            % Plot resutls
            %
            % Parameters
            % ----------
            % - props_filename: char vector
            %   Path of `properties`

            % parameters
            % - time resolution
            viz = DagNNViz();
            dt_sec = Neda.dt_sec;
            
            % main
            DagNNViz.print_title(props_filename);
            % props
            props = jsondecode(fileread(props_filename));
            % 'bak' dir
            bak_dir = props.data.bak_dir;

            % output dir
            % - path
            viz.output_dir = fullfile(bak_dir, 'images');
            % - make if doesn't exist
            if ~exist(viz.output_dir, 'dir')
                mkdir(viz.output_dir);
            end

            % costs
            % - make
            costs = load(fullfile(bak_dir, 'costs.mat'));
            % - plot
            viz.plot_costs(costs);
            
            % net
            viz.plot_digraph(props_filename);

            % params
            viz.plot_params(...
                {props.net.params.name}, ...
                bak_dir, ...
                props.learning.number_of_epochs + 1, ...
                dt_sec ...
            );
        
            % db
            % - all
            db = load(props.data.db_filename);
            small_db_size = 50;
            viz.plot_db_all(...
                db, ...
                small_db_size ...
            );
            % - first
            viz.plot_db_first(...
                db, ...
                dt_sec ...
            );
        end
    end
end
