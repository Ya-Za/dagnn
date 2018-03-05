classdef Viz < handle
    %Visualize a 'DagNN'
    
    % Filename
    properties (Constant)
        % Properties
        % ----------
        % - DATA_FILENAME: char vector
        %   Data filename
        % - CONFIG_FILENAME: char vector
        %   Config filename
        % - DATA_INDEXES_FILENAME: char vector
        %   Data indexes filename
        % - COSTS_FILENAME: char vector
        %   Costs filename
        
        DATA_FILENAME = 'data.mat';
        CONFIG_FILENAME = 'config.json';
        DATA_INDEXES_FILENAME = 'data-indexes.mat';
        COSTS_FILENAME = 'costs.mat';
        VIDEOS_DIR = 'vidoes';
        PARAMS_EXPECTED_FILENAME = 'params-expected';
        PARAMS_INITIAL_FILENAME = 'params-initial';
        EPOCHS_DIR = 'epochs';
    end
    
    % Plot
    properties (Constant)
        % Properties
        % ----------
        % - DT: double
        %   Time resolution in seconds
        % - STIM_COLOR: color
        %   Color of stimulus
        % - RESP_COLOR: color
        %   Color of response
        % - XLABEL: char vector
        %   X label is same for `stimului` and `responses`
        % - STIM_YLABEL: char vector
        %   Y label of stimulus
        % - RESP_YLABEL: char vector
        %   Y label of response
        % - STIM_TITLE: char vector
        %   Title of stimulus
        % - RESP_TITLE: char vector
        %   Title of response
        % - ROUND_DIGITS: int
        %   Rounds to N digits
        
        DT = 0.001;
        STIM_COLOR = [0, 0.4470, 0.7410];
        RESP_COLOR = [0.8500, 0.3250, 0.0980];
        TRAIN_COLOR = [0, 0.4470, 0.7410];
        VAL_COLOR = [0.4660, 0.6740, 0.1880];
        TEST_COLOR = [0.8500, 0.3250, 0.0980];
        NO_COLOR = [1, 1, 1];
        XLABEL = 'Time (s)';
        STIM_YLABEL = 'Intensity';
        RESP_YLABEL = 'Rate (Hz)';
        STIM_TITLE = 'Stimulus';
        RESP_TITLE = 'Response';
        ROUND_DIGITS = 3;
    end
    
    properties
        % Properties
        % ----------
        % - X: cell array
        %   Stimulus set
        % - Y: cell array
        %   Response set
        % - N: int
        %   Number of sample
        % - stimulus: double array
        %   Concatenate all stimuli
        % - response: double array
        %   Concatenate all responses
        % - learningParams: struct(...
        %       'trainValTestRatios', [double, double, double], ...
        %       'learningRate', double, ...
        %       'batchSize', int, ...
        %       'numberOfEpochs', int, ...
        %   )
        %   Learning parameters
        % - dataIndexes: struct(...
        %       'train', int vector, ...
        %       'val', int vector, ...
        %       'test', int vector ...
        %   )
        %   Contains 'train', 'val' and 'test' indexes
        % - costs: struct(...
        %       'train', double vector, ...
        %       'val', double vector, ...
        %       'test', double vector ...
        %   )
        %   Contains 'train', 'val' and 'test' costs
        
        path
        X
        Y
        N
        stimulus
        response
        learningParams
        dataIndexes
        costs
        paramNames
        params
    end
    
    % Constructor
    methods
        function obj = Viz(path)
            % Constructor
            %
            % Parameters
            % ----------
            % - path: char vector
            %   Path of data directory
            
            obj.path = path;
            
            % data
            data = load(fullfile(path, Viz.DATA_FILENAME));
            obj.X = data.x;
            obj.Y = data.y;
            
            obj.N = min(length(obj.X), length(obj.Y));
            
            obj.stimulus = vertcat(obj.X{:});
            obj.response = vertcat(obj.Y{:});
            
            obj.initLearningParams(path);
            obj.initDataIndexes(path);
            
            obj.costs = load(fullfile(path, Viz.COSTS_FILENAME));
            
            obj.initParamNames(path);
            obj.initParams(path);
        end
        function initLearningParams(obj, path)
            % Init `learningParams` property
            %
            % Parameters
            % ----------
            % - path: char vector
            %   Path of data directory
            
            % load config file
            config = jsondecode(fileread(fullfile(path, Viz.CONFIG_FILENAME)));
            
            % learning parameters
            obj.learningParams = struct(...
                'trainValTestRatios', config.learning.train_val_test_ratios, ...
                'learningRate', config.learning.learning_rate, ...
                'batchSize', config.learning.batch_size, ...
                'numberOfEpochs', config.learning.number_of_epochs ...
            );
        end
        function initDataIndexes(obj, path)
            % Init `dataIndexes` property
            %
            % Parameters
            % ----------
            % - path: char vector
            %   Path of data directory
            
            % load shuffled data indexes
            obj.dataIndexes = load(...
                fullfile(path, Viz.DATA_INDEXES_FILENAME) ...
            );
        end
        function initParamNames(obj, path)
            % load config file
            config = jsondecode(fileread(fullfile(path, Viz.CONFIG_FILENAME)));
            
            % parameters names
            obj.paramNames = {config.net.params.name};
        end
        function initParams(obj, path)
            initParmsExpected();
            initParamsInitial();
            initParamsHistory();
            
            % Local Functions
            function initParmsExpected()
                paramsExpected = load(fullfile(path, Viz.PARAMS_EXPECTED_FILENAME));
                
                for i = 1:length(obj.paramNames)
                    paramName = obj.paramNames{i};
                    obj.params.(paramName).expected = paramsExpected.(paramName);
                end
            end
            function initParamsInitial()
                paramsInitial = load(fullfile(path, Viz.PARAMS_INITIAL_FILENAME));
                
                for i = 1:length(obj.paramNames)
                    paramName = obj.paramNames{i};
                    obj.params.(paramName).initial = paramsInitial.(paramName);
                end
            end
            function initParamsHistory()
                % filenames of saved 'epoch' files
                epochsDir = fullfile(path, Viz.EPOCHS_DIR);
                filenames = dir(fullfile(epochsDir, '*.mat'));
                filenames = {filenames.name};
                
                paramIndexes = Viz.getParamIndexes(fullfile(epochsDir, filenames{1}));

                % param-hsitory
                for i = 1:length(filenames)
                     epochParams = getfield(...
                        load(fullfile(epochsDir, filenames{i})), ...
                        'params' ...
                    ); 
                    for j = 1:length(obj.paramNames)
                        paramName = obj.paramNames{j};
                        obj.params.(paramName).history{i} = ...
                            epochParams(paramIndexes.(paramName)).value;
                    end
                end
            end
        end
    end
    
    % Plot Data
    methods
        function plotData(obj, numberOfSamples)
            % Plot first stim/resp samples in a grid pattern
            %
            % Parameters
            % ----------
            % - numberOfSamples: int = 50
            %   Number of samples (stim/res)
            
            % properties
            fontsize = 6;
            
            % plot limits
            xMin = 1;
            % todo: change for data with vraible length
            xMax = max(length(obj.X{1}), length(obj.Y{1}));
            yMin = min(min(obj.stimulus), min(obj.response));
            yMax = max(max(obj.stimulus), max(obj.response));
            limits = [xMin, xMax, yMin, yMax];
            
            % figure
            Viz.figure('Data');
        
            % number of samples
            if ~exist('numberOfSamples', 'var')
                numberOfSamples = min(50, obj.N);
            else
                numberOfSamples = min(numberOfSamples, obj.N);
            end
            
            % subplot grid
            [rows, cols, numberOfSamples] = getColsRows(numberOfSamples);
            
            % plot
            % - first stim/resp pair
            %   - stimulus
            plotFirstStimulus();
            %   - response
            plotFirstResponse();
            
            % - other samples
            h = waitbar(0, 'Plot data...');
            for sampleIndex = 2:numberOfSamples
                % - stimulus
                plotStimulus(sampleIndex);
                % - response
                plotResponse(sampleIndex);
                
                waitbar(sampleIndex / numberOfSamples)
            end
            close(h)
            
            % super-title
            suptitle(...
                sprintf(...
                    'First %d of %d Paird (Stimulous, Response) Samples', ...
                    numberOfSamples, ...
                    obj.N ...
                ) ...
            );
        
            % Local Functions
            function [rows, cols, n] = getColsRows(n)
                % number of plots = 2 * number of samples(stim/resp)
                n = 2 * n;
                % - cols
                cols = floor(sqrt(n));
                % - cols must be even
                if mod(cols, 2) == 1
                    cols = cols + 1;
                end
                % - rows
                rows = floor(n / cols);
                % - n
                n = (rows * cols) / 2;
            end
            function setAxis()
                axis(limits);
                set(gca, 'XAxisLocation', 'origin');
                Viz.hideticks();
            end
            function plotStimulus(i)
                subplot(rows, cols, 2 * (i - 1) + 1);
                plot(obj.X{i}, 'Color', Viz.STIM_COLOR);
                title(num2str(i));
                setTinyFontSize();
                
                setAxis();
            end
            function plotResponse(i)
                subplot(rows, cols, 2 * i);
                plot(obj.Y{i}, 'Color', Viz.RESP_COLOR);
                
                setAxis();
            end
            function plotFirstStimulus()
                plotStimulus(1);
                
                title(Viz.STIM_TITLE);
                xlabel(Viz.XLABEL);
                ylabel(Viz.STIM_YLABEL);
                
                setTinyFontSize();
            end
            function plotFirstResponse()
                plotResponse(1);
                
                title(Viz.RESP_TITLE);
                xlabel(Viz.XLABEL);
                ylabel(Viz.RESP_YLABEL);
                
                setTinyFontSize();
            end
            function setTinyFontSize()
                set(gca, 'FontSize', fontsize);
            end
        end
        function plotSample(obj, sampleNumber)
            % Plot sample (stim/resp)
            %
            % Parameters
            % ----------
            % - sampleNumer: int = 1
            %   number of target sample
            
            % default values
            if (~exist('sampleNumber', 'var'))
                sampleNumber = 1;
            end
            
            % figure
            Viz.figure(...
                sprintf('Data - Sample #%d', sampleNumber) ...
            );
        
            % data
            % - stimulus
            stimValues = obj.X{sampleNumber};
            stimTimes = ((1:length(stimValues)) - 1) * Viz.DT;
            % - response
            respValues = obj.Y{sampleNumber};
            respTimes = ((1:length(respValues)) - 1) * Viz.DT;
            % - limits
            limits = [
                0
                max(max(stimTimes), max(respTimes))
                min(min(stimValues), min(respValues))
                max(max(stimValues), max(stimValues)) + 0.01
            ];
            
            % plot
            % - stimulus
            subplot(121);
            Viz.plot(...
                stimTimes, ...
                stimValues, ...
                Viz.STIM_COLOR, ...
                Viz.STIM_TITLE, ...
                Viz.XLABEL, ...
                Viz.STIM_YLABEL ...
            );
            setAxis();
            
            % - response
            subplot(122);
            Viz.plot(...
                respTimes, ...
                respValues, ...
                Viz.RESP_COLOR, ...
                Viz.RESP_TITLE, ...
                Viz.XLABEL, ...
                Viz.RESP_YLABEL ...
            );
            setAxis();
            
            % - super-title
            suptitle(...
                sprintf(...
                    'Sample (Stimulous, Response) #%d ', ...
                    sampleNumber ...
                ) ...
            );        
            
            % Local Functions
            function setAxis()
                axis(limits);
                set(gca, 'XAxisLocation', 'origin');
            end
        end
        function boxplotData(obj)
            % Boxplot of data
            
            % figure
            Viz.figure('Box Plot - Data');
            
            % stimulus
            subplot(121);
            boxplot(...
                obj.stimulus, ...
                'Notch', 'off', ...
                'Labels', {Viz.STIM_TITLE} ...
            );
            ylabel(Viz.STIM_YLABEL);
            
            % response
            subplot(122);
            boxplot(...
                obj.response, ...
                'Notch', 'off', ...
                'Labels', {Viz.RESP_TITLE} ...
            );
            ylabel(Viz.RESP_YLABEL);
            
            % title
            suptitle('Box Plot of Data (Stimulus, Response)');
        end
        function boardplotData(obj)
            % Board plot of data
            
            % figure
            Viz.figure('Board Plot - Data');
            
            % board
            cols = floor(sqrt(obj.N));
            rows = ceil(obj.N / cols);
            B = 4 * ones(cols, rows); % must be transpose
            
            % train
            B(obj.dataIndexes.train) = 1;
            % val
            B(obj.dataIndexes.val) = 2;
            % test
            B(obj.dataIndexes.test) = 3;
            
            % colors
            colors = [
                Viz.TRAIN_COLOR
                Viz.VAL_COLOR
                Viz.TEST_COLOR
                % Viz.NO_COLOR
            ];

            % plot
            pcolor(B');
            % imagesc(B');
            % colormap
            colormap(colors)
            % title
            r = floor(100 * obj.learningParams.trainValTestRatios);
            title(...
                sprintf('Train (%d%%) | Validation (%d%%) | Test (%d%%)', ...
                r(1), r(2), r(3)) ...
            );
            % axis
            axis('ij');
            axis('square');
            set(gca, ...
                'XAxisLocation', 'top', ...
                'XTick', [], ...
                'YTick', [] ...
            );
            % colorbar
            colorbar(...
                'Location', 'southoutside', ...
                'Ticks', [] ...
            );
        end
    end
    
    % Plot Errors
    methods
        function plotCosts(obj, order, yScale)
            % Plot 'costs' over time  
            %
            % Parameters
            % ----------
            % - order: [] (default) | 'diff'
            %   First difference - Approximate first derivatives
            % - yScale: 'log' (default) | 'linear'
            %   Scale of values along y-axis  
            
            setDefaultValues();
            
            % Local Parameters
            % ----------------
            % - train: double vector
            %   Train costs
            % - val: double vector
            %   Validation costs
            % - test: double vector
            %   Test costs
            % - epochs: int vector
            %   Epoch numbers
            % - cx: int
            %   Epoch number with minimum validation cost
            % - cy: double
            %   Minimum validation cost
            train = [];
            val = [];
            test = [];
            epochs = [];
            cx = [];
            cy = [];
            setLocalParameters();
            
            % plots
            plotErrors();
            drawCircleAndCrosslines();
            setAxes();
            
            % Local Functions
            function setDefaultValues()
                % default values
                if ~exist('order', 'var')
                    order = [];
                end
                if ~exist('yScale', 'var')
                    yScale = 'log';
                end
            end
            function setLocalParameters()
                % train, validation and test costs
                train = obj.costs.train;
                val = obj.costs.val;
                test = obj.costs.test;
                if ~isempty(order)
                    train = diff(train);
                    val = diff(val);
                    test = diff(test);
                end

                % epochs
                epochs = 1:length(train);
                epochs = epochs - 1; % start epochs from zero (0, 1, 2, ...)

                % minimum validation error
                [cy, cx] = min(val);
            end
            function plotErrors()
                lineWidth = 2;
                % figure
                Viz.figure('CNN - Costs [Training, Validation, Test]');
                % - train
                plot(epochs, train, ...
                    'LineWidth', lineWidth, ...
                    'Color', Viz.TRAIN_COLOR ...
                );
                set(gca, 'YScale', yScale);
                hold('on');
                % - validation
                plot(epochs, val, ...
                    'LineWidth', lineWidth, ...
                    'Color', Viz.VAL_COLOR ...
                );
                % - test
                plot(epochs, test, ...
                    'LineWidth', lineWidth, ...
                    'Color', Viz.TEST_COLOR ...
                );
                hold('off');
            end
            function drawCircleAndCrosslines()
                color = [0.1, 0.8, 0.1];
                color = [0.9290, 0.6940, 0.1250];
                lineWidth = 1.5;
                hold('on');
                % - circle
                drawCircle(cx - 1, cy);
                % - cross cross lines
                drawHorizontalLine(cy);
                drawVerticalLine(cx - 1);
                hold('off');
                
                % Local Functions
                function drawCircle(cx, cy)
                    circleSize = 75;
                    scatter(cx - 1, cy, ...
                        'MarkerEdgeColor', color, ...
                        'SizeData', circleSize, ...
                        'LineWidth', lineWidth ...
                    );
                end
                function drawHorizontalLine(y)
                    ax = gca;
                    line(...
                        ax.XLim, ...
                        [y, y], ...
                        'Color', color, ...
                        'LineStyle', ':', ...
                        'LineWidth', lineWidth ...
                    );
                end
                function drawVerticalLine(x)
                    ax = gca;
                    line(...
                        [x, x], ...
                        ax.YLim, ...
                        'Color', color, ...
                        'LineStyle', ':', ...
                        'LineWidth', lineWidth ...
                    );
                end
            end
            function setAxes()
                setTicks();
                % setAxesLocations();
                setLabels();
                setTitle();
                setLegend();
                grid('on');
                box('off');
                
                % Local Functions
                function setTicks()
                    % ticks
                    % - x
                    set(gca, ...
                        'XTick', unique([0, cx - 1, epochs(end)]) ...
                    );
                    % - y
                    set(gca, ...
                        'YTick', ...
                        unique(sort([...
                            0, ...
                            round(cy, Viz.ROUND_DIGITS), ...
                            round(max([train, val, test]), Viz.ROUND_DIGITS) ...
                        ])) ...
                    );
                end
                function setAxesLocations()
                    set(gca, 'XAxisLocation', 'origin');
                    set(gca, 'YAxisLocation', 'origin');
                end
                function setLabels()
                    % labels
                    xlabel('Epoch');
                    if isempty(order)
                        prefix = '';
                    else
                        prefix = 'Frist Derivatives of the ';
                    end
                    ylabel([prefix, 'Mean Squared Error (Hz^2)']);
                end
                function setTitle()
                    % title
                    if isempty(order)
                        prefix = 'Minimum';
                    else
                        prefix = 'Maximum Difference of';
                    end
                    title(...
                        sprintf('%s Validation Error is %g at Epoch: %d', ...
                            prefix, ...
                            cy, ...
                            cx - 1 ...
                        ) ...
                    );
                end
                function setLegend()
                    % legend
                    legend(...
                        sprintf('Training (%g)', train(cx)), ...
                        sprintf('Validation (%g)', val(cx)), ...
                        sprintf('Test (%g)', test(cx)), ...
                        'Best Validation Error' ...
                    );
                end
            end
        end
    end
    
    % Plot Filters
    methods
        function playFilterVideo(obj, filterName)
            % Play or save video of learning process of the specific filter
            
            filterVideoFilename = obj.getFilterVideoFilename(filterName);
            if ~exist(filterVideoFilename, 'file')
                obj.saveFilterVideo(filterName, filterVideoFilename);
            end
            
            Viz.playVideo(filterVideoFilename);
        end
        function filterVideoFilename = getFilterVideoFilename(obj, filterName)
            filterVideoFilename = fullfile(...
                obj.path, ...
                Viz.VIDEOS_DIR, ...
                [filterName, '.mp4'] ...
            );
        end
        function saveFilterVideo(obj, filterName, filterVideoFilename)
            
            % Parameters
            frameRate = 10;
            delay = 0.01;
            lineWidth = 2;
            filterExpectedColor = [0.4660, 0.6740, 0.1880];
            filterInitialColor = [0.8500, 0.3250, 0.0980];
            filterMinErrorColor = [0.9290, 0.6940, 0.1250];
            filterHistoryColor = [0, 0.4470, 0.7410];
            limits = obj.getFilterLimits(filterName);
            
            % open video writer
            vw = Viz.openVideoWriter(filterVideoFilename, frameRate);

            fig = Viz.figure(sprintf('Filter: %s', filterName));
            ax = gca;
            
            plotFilterExpected();
            hold('on');
            plotFilterInitial();
            plotFilterMinError();
            
            saveFrame();
            
            % initial epoch
            plotFilterHistory(1);
            saveFrame();
            
            numberOfEpochs = length(obj.params.(filterName).history);
            for i = 2:numberOfEpochs
                updatePlotFilterHistory(i);
                saveFrame();
                
                pause(delay);
            end
            
            % close video writer
            close(vw);
            
            % Local Functions
            function saveFrame()
                setAxes();
                writeVideo(vw, getframe(fig));
            end
            function setAxes()
                setTicks();
                % setAxesLocations();
                % setLabels();
                setLegend();
                grid('on');
                box('off');
                
                % Local Functions
                function setTicks()
                    axis(limits);
                    set(gca, 'XTick', limits(1:2));
                    set(gca, 'YTick', unique(sort([limits(3), 0, limits(4)])));
                end
                function setAxesLocations()
                    set(gca, 'XAxisLocation', 'origin');
                    set(gca, 'YAxisLocation', 'origin');
                end
                function setLabels()
                    xlabel('#');
                    ylabel('Intensity');
                end
                function setLegend()
                    % legend
                    lgd = legend('show');
                    lgd.Location = 'southwest';
                    lgd.Box = 'off';
                end
            end
            function plotFilterExpected()
                plot(obj.params.(filterName).expected, ...
                    'DisplayName', 'Ground truth', ...
                    'LineWidth', lineWidth, ...
                    'Color', filterExpectedColor ...
                );
            end
            function plotFilterInitial()
                plot(obj.params.(filterName).initial, ...
                    'DisplayName', 'Initial value', ...
                    'LineWidth', lineWidth, ...
                    'Color', filterInitialColor ...
                );
            end
            function plotFilterMinError()
                [~, index] = min(obj.costs.val);
                plot(obj.params.(filterName).history{index}, ...
                    'DisplayName', sprintf('Min Val Error (#%d)', index - 1), ...
                    'LineStyle', '--', ...
                    'LineWidth', lineWidth - 0.5, ...
                    'Color', filterMinErrorColor ...
                );
            end
            function plotFilterHistory(index)
                plot(obj.params.(filterName).history{index}, ...
                    'DisplayName', sprintf('Current value (#%d)', index - 1), ...
                    'LineWidth', lineWidth, ...
                    'Color', filterHistoryColor ...
                );
                setTitle(index);
            end 
            function updatePlotFilterHistory(index)
                ax.Children(1).YData = obj.params.(filterName).history{index};
                ax.Children(1).DisplayName = sprintf('Current value (#%d)', index - 1);
                setTitle(index);
            end
            function setTitle(index)
                title(sprintf(...
                    'Epoch: %d, Error: (%g, %g, %g)', ...
                    index - 1, ...
                    obj.costs.train(index), ...
                    obj.costs.val(index), ...
                    obj.costs.test(index) ...
                ));
            end
        end
        function limits = getFilterLimits(obj, filterName)
            % xMin
            xMin = 1;
            % xMax
            xMax = length(obj.params.(filterName).expected);
            % yMin, yMax
            yMin = Inf;
            yMax = -Inf;
            % - expected
            yMin = min(yMin, min(obj.params.(filterName).expected));
            yMax = max(yMax, max(obj.params.(filterName).expected));
            % - initial
            yMin = min(yMin, min(obj.params.(filterName).initial));
            yMax = max(yMax, max(obj.params.(filterName).initial));
            % - history
            numberOfEpochs = length(obj.params.(filterName).history);
            for i = 1:numberOfEpochs
                yMin = min(yMin, min(obj.params.(filterName).history{i}));
                yMax = max(yMax, max(obj.params.(filterName).history{i}));
            end
            
            limits = [xMin, xMax, yMin, yMax];
        end
    end
        
    % Utils
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
                'Color', 'white', ...
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
        function d = getRoundDigits(xMin, xMax)
            % Get first number of digits, that distinguish between 
            % `xMin` and `xMax`
            %
            % Parameters
            % ----------
            % - xMin: double
            %   Begin of interval
            % - xMax: double
            %   End of interval
            %
            % Returns
            % -------
            % - d: int
            %   Number of digits to the right of the decimal point
            
            d = Viz.ROUND_DIGITS;
            xMin = round(xMin, d);
            xMax = round(xMax, d);
            
            while xMin == xMax
                d = d + 1;
                xMin = round(xMin, d);
                xMax = round(xMax, d);
            end
        end
        function twoticks(x, axisName)
            % Show just `min` and `max` of ticks
            %
            % Parameters
            % ----------
            % - x: double vector
            %   Input values
            % - axisName: cahr vector
            %   'XTick' or 'YTick' 
            
            xMin = min(x);
            xMax = max(x);
            
            d = Viz.getRoundDigits(xMin, xMax);
            
            xMin = round(xMin, d);
            xMax = round(xMax, d);
            
            set(gca, axisName, [xMin, xMax]);
        end
        function plot(x, y, color, titleTxt, xlabelTxt, ylabelTxt)
            % Like as matlab `plot`
            %
            % Parameters
            % ----------
            % - x: double vector
            %   `x` input
            % - y: double vector
            %   `y` input
            % - color: char vector
            %   Color of plot
            % - titleTxt: char vector
            %   Title of plot
            % - xlabelTxt: char vector
            %   Label of `x` axis
            % - ylabelTxt: char vector
            %   Label of `y` axis

            plot(x, y, 'Color', color);
            
            title(titleTxt);
            xlabel(xlabelTxt);
            ylabel(ylabelTxt);
            
            Viz.twoticks(x, 'XTick');
            Viz.twoticks(y, 'YTick');
            
            grid('on');
            box('off');
        end
        function playVideo(filename)
            implay(filename);
        end
        function paramIndexes = getParamIndexes(filename)
            epochParams = getfield(load(filename), 'params'); 
            for i = 1:length(epochParams)
                paramIndexes.(epochParams(i).name) = i;
            end
        end
        function vw = openVideoWriter(filename, frameRate)
            [folderName, ~, ~] = fileparts(filename);
            if ~exist(folderName, 'dir')
                mkdir(folderName)
            end
            
            vw = VideoWriter(filename, 'MPEG-4');
            vw.FrameRate = frameRate;
            open(vw);
        end
    end
    
end
