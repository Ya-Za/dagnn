classdef Viz < handle
    %Visualize a 'DagNN'
    
    % Plot Constants
    properties (Constant)
        % Properties
        % ----------
        % - DT: double
        %   Time resolution in seconds
        % - STIM_COLOR: color
        %   Color of stimulus
        % - RESP_COLOR: color
        %   Color of response
        % - TRAIN_COLOR: color
        %   Color of training set
        % - VAL_COLOR: color
        %   Color of validation set
        % - TEST_COLOR: color
        %   Color of test set
        % - NO_COLOR: color
        %   Color of nothing!
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
        % - FORMATTYPE: char vector
        %   File format such as `fig`, `epsc`, `pdf`, `svg`, `png` or ...
        
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
        FORMATTYPE = 'fig';
    end
    
    % Data
    properties
        % Properties
        % ----------
        % - path: char vector
        %   Path of root directory
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
        % - corr: struct(...
        %       'train', double vector, ...
        %       'val', double vector, ...
        %       'test', double vector ...
        %   )
        %   Contains 'train', 'val' and 'test' correlation
        % - paramNames: cell array
        %   Name of parameters
        % - filterNames: cell array
        %   Name of filters
        % - biasNames: cell array
        %   Name of bias parameters
        % - params: struct(...
        %       'name', struct(...
        %           'expected', double array, ...
        %           'initial', double array, ...
        %           'history', cell array ...
        %   )
        %   Expected, initial and history of parameters
        % - dag: DagNNTrainer
        %   Directed acyclic graph network
        
        path
        config
        X
        Y
        N
        numOfEpochs
        stimulus
        response
        learningParams
        dataIndexes
        costs
        corr
        paramNames
        filterNames
        biasNames
        params
        dag
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
            
            % load config file
            obj.config = jsondecode(fileread(fullfile(obj.path, Path.CONFIG_FILENAME)));
            
            % data
            data = load(fullfile(path, Path.DATA_FILENAME));
            obj.X = data.x;
            obj.Y = data.y;
            
            obj.N = min(length(obj.X), length(obj.Y));
            obj.initNumOfEpochs();
            
            obj.stimulus = vertcat(obj.X{:});
            obj.response = vertcat(obj.Y{:});
            
            obj.initLearningParams();
            obj.initDataIndexes();
            
            obj.costs = load(fullfile(path, Path.COSTS_FILENAME));
            
            obj.initParamNames();
            obj.initFilterNames();
            obj.initBiasNames();
            obj.initParams();
            obj.initDag();
            obj.initFigDir();
            
            obj.initCorr();
        end
        function initNumOfEpochs(obj)
            epochsDir = fullfile(obj.path, Path.EPOCHS_DIR);
            filenames = dir(fullfile(epochsDir, '*.mat'));
            obj.numOfEpochs = length(filenames);
        end
        function initLearningParams(obj)
            % Init `learningParams` property
            %
            % Parameters
            % ----------
            % - path: char vector
            %   Path of data directory
            
            % learning parameters
            obj.learningParams = struct(...
                'trainValTestRatios', obj.config.learning.train_val_test_ratios, ...
                'learningRate', obj.config.learning.learning_rate, ...
                'batchSize', obj.config.learning.batch_size, ...
                'numberOfEpochs', obj.config.learning.number_of_epochs ...
            );
        end
        function initDataIndexes(obj)
            % Init `dataIndexes` property
            %
            % Parameters
            % ----------
            % - path: char vector
            %   Path of data directory
            
            % load shuffled data indexes
            obj.dataIndexes = struct();
            indexes = load(...
                fullfile(obj.path, Path.DATA_INDEXES_FILENAME) ...
            );
            indexes = indexes.db_indexes;
        
            % number of samples
            n = obj.N;
            
            % ratios
            ratios = obj.learningParams.trainValTestRatios;
            ratios = ratios / sum(ratios);
            
            % end index
            % - train
            endIndexTrain = floor(ratios(1) * n);
            % - val
            endIndexVal = floor((ratios(1) + ratios(2)) * n);
            % - test
            endIndexTest = n;
            
            % data
            % - train
            obj.dataIndexes.train = sort(indexes(1:endIndexTrain));
            
            % - val
            obj.dataIndexes.val = sort(indexes(endIndexTrain + 1:endIndexVal));
            
            % - test
            obj.dataIndexes.test = sort(indexes(endIndexVal + 1:endIndexTest));
        end
        function initParamNames(obj)
            % parameters names
            obj.paramNames = {obj.config.net.params.name};
        end
        function initFilterNames(obj)
            parameters = obj.config.net.params;
            obj.filterNames = {};
            for i = 1:length(parameters)
                % bias names has only one element
                if prod(parameters(i).size) > 1
                    obj.filterNames{end + 1} = parameters(i).name;
                end
            end
        end
        function initBiasNames(obj)
            parameters = obj.config.net.params;
            obj.biasNames = {};
            for i = 1:length(parameters)
                % bias names has only one element
                if prod(parameters(i).size) == 1
                    obj.biasNames{end + 1} = parameters(i).name;
                end
            end
        end
        function initParams(obj)
            initParmsExpected();
            initParamsInitial();
            initParamsHistory();
            initMinValCost();
            
            % Local Functions
            function initParmsExpected()
                paramsExpected = load(fullfile(obj.path, Path.PARAMS_EXPECTED_FILENAME));
                
                for i = 1:length(obj.paramNames)
                    paramName = obj.paramNames{i};
                    obj.params.(paramName).expected = paramsExpected.(paramName);
                end
            end
            function initParamsInitial()
                paramsInitial = load(fullfile(obj.path, Path.PARAMS_INITIAL_FILENAME));
                
                for i = 1:length(obj.paramNames)
                    paramName = obj.paramNames{i};
                    obj.params.(paramName).initial = paramsInitial.(paramName);
                end
            end
            function initParamsHistory()
                % filenames of saved 'epoch' files
                epochsDir = fullfile(obj.path, Path.EPOCHS_DIR);
                filenames = dir(fullfile(epochsDir, '*.mat'));
                filenames = {filenames.name};
                
                % sort filenames based on number pattern
                filenums = cellfun(@(x)sscanf(x,'%d.mat'), filenames);
                [~, I] = sort(filenums);
                filenames = filenames(I);
                
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
            function initMinValCost()
                for i = 1:length(obj.paramNames)
                    paramName = obj.paramNames{i};
                    
                    [value, index] = min(obj.costs.val);
                    obj.params.(paramName).minValCost = struct(...
                        'index', index, ...
                        'value', value ...
                    );
                end
            end
        end
        function initDag(obj)
            run('vl_setupnn.m');
            obj.dag = DagNNTrainer(fullfile(obj.path, Path.CONFIG_FILENAME));
            obj.dag.init();
            [~, epoch] = min(obj.costs.val);
            obj.dag.load_epoch(epoch);
        end
        function initFigDir(obj)
            folder = fullfile(obj.path, Path.FIGURES_DIR);
            if ~exist(folder, 'dir')
                mkdir(folder)
            end
        end
        function initCorr(obj)
            [train, val, test] = obj.error(@(u, v) corr(u, v));
            obj.corr.train = train;
            obj.corr.val = val;
            obj.corr.test = test;
        end
    end
    
    % Draw Network
    methods
        function dg = makeDigraph(obj)
            % Make a directed-graph based on `config.json` file
            %
            % Returns
            % - dg : digraph
            %   Directed graph
            
            layers = obj.config.net.layers';
            
            % add layers to digraph
            dg = digraph();
            for layer = layers
                block = sprintf('%s(%s)', layer.name, layer.type);
                % block = layer.type;
                
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
        function nodes = getNodes(obj)
            net = obj.config.net;
            nodes = struct();
            
            % input
            nodes.input = net.vars.input.name;
            % output
            nodes.output = net.vars.output.name;
            % expected output
            nodes.expectedOutput = net.vars.expected_output.name;
            % cose
            nodes.cost = net.vars.cost.name;
            % params
            nodes.params = {net.params.name};
            % layers, blocks
            nodes.layers = {};
            nodes.blocks = {};
            for layer = net.layers'
                nodes.layers = {nodes.layers{:}, layer.outputs{:}};
                nodes.blocks{end + 1} = sprintf('%s(%s)', layer.name, layer.type);
            end
        end
        function plotNet(obj)
            % Plot a directed-graph based on `config.json` file
            
            squareSize = 50;
            circleSize = 7;
            fontSize = 11;
            dataColor = [0.8500, 0.3250, 0.0980];
            paramsColor = [0.4660, 0.6740, 0.1880];
            layersColor = [0, 0.4470, 0.7410];
            outputColor = [0.6350, 0.0780, 0.1840];
            costColor = [0.4940, 0.1840, 0.5560];
            
            % make digraph
            dg = obj.makeDigraph();
            
            % nodes
            nodes = obj.getNodes();
            
            % figure
            Viz.figure('Net');
        
            % node labels
            labels = {};
            %   name(type) -> type
            expression = '\w+\((?<type>[\+\w]+)\)';
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

            % title
            title('Structure of Network');
            
            % layout
            layout(h, 'layered', ...
                'Direction', 'right', ...
                'Sources', nodes.input, ...
                'Sinks', nodes.cost, ...
                'AssignLayers', 'asap' ...
            );
        
            % font
%             nl = h.NodeLabel;
%             h.NodeLabel = '';
%             xData = get(h, 'XData');
%             yData = get(h, 'YData');
%             yData = yData - 0.2;
%             text(...
%                 xData, ...
%                 yData, ...
%                 nl, ...
%                 'FontSize', fontSize, ...
%                 'FontWeight', 'bold', ...
%                 'HorizontalAlignment', 'center', ...
%                 'VerticalAlignment', 'top' ...
%             );
            
            % highlight
            % - data = (input, expected output)
            highlight(h, ...
                {nodes.input, nodes.expectedOutput}, ...
                'NodeColor', dataColor, ...
                'MarkerSize', circleSize ...
            );
            % - parameters
            highlight(h, ...
                nodes.params, ...
                'NodeColor', paramsColor, ...
                'MarkerSize', circleSize ...
            );
            % - layers
            highlight(h, ...
                nodes.layers, ...
                'NodeColor', layersColor, ...
                'MarkerSize', circleSize ...
            );
            % - blocks
            highlight(h, ...
                nodes.blocks, ...
                'Marker', 's', ...
                'MarkerSize', squareSize ...
            );
            % - output
            highlight(h, ...
                nodes.output, ...
                'NodeColor', outputColor, ...
                'MarkerSize', circleSize ...
            );
            % - cost
            highlight(h, ...
                nodes.cost, ...
                'NodeColor', costColor, ...
                'MarkerSize', circleSize ...
            );

            % hide axes
            set(h.Parent, ...
                'XTick', [], ...
                'YTick', [] ...
            );
        
            % save
            obj.saveFigure('net');
        end
    end
    
    % Plot Data
    methods
        function plotData(obj, indexes)
            % Plot first stim/resp samples in a grid pattern
            %
            % Parameters
            % ----------
            % - numberOfSamples: int = 50
            %   Number of samples (stim/res)
            
            % plot each `n` epochs in a new figure
            if ~exist('indexes', 'var')
                plotDataAll();
                
                return;
            end
            
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
            numberOfSamples = length(indexes);
            
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
                    '%d of %d Paird (Stimulous, Response) Samples', ...
                    numberOfSamples, ...
                    obj.N ...
                ) ...
            );
        
            % save
            obj.saveFigure('data');
        
            % Local Functions
            function plotDataAll()
                n = 50; % number of epochs in each figure
                s = 1; % stard epoch
                f = n; % finish epoch
                
                while s < obj.N
                    if f > obj.N
                        f = obj.N;
                    end
                    
                    obj.plotData(s:f);
                    
                    s = s + n;
                    f = f + n;
                end
            end
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
                
                ax = gca;
                ax.YAxis.Visible = 'off';
            end
            function plotStimulus(i)
                subplot(rows, cols, 2 * (i - 1) + 1);
                plot(obj.X{indexes(i)}, 'Color', Viz.STIM_COLOR);
                title(num2str(indexes(i)));
                setTinyFontSize();
                
                setAxis();
            end
            function plotResponse(i)
                subplot(rows, cols, 2 * i);
                plot(obj.Y{indexes(i)}, 'Color', Viz.RESP_COLOR);
                
                setAxis();
            end
            function plotFirstStimulus()
                plotStimulus(1);
                
                title(Viz.STIM_TITLE);
                xlabel(Viz.XLABEL);
                ylabel(Viz.STIM_YLABEL);
                
                ax = gca;
                ax.YAxis.Visible = 'on';
                
                setTinyFontSize();
            end
            function plotFirstResponse()
                plotResponse(1);
                
                title(Viz.RESP_TITLE);
                xlabel(Viz.XLABEL);
                ylabel(Viz.RESP_YLABEL);
                
                ax = gca;
                ax.YAxis.Visible = 'on';
                
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
            
            % save
            obj.saveFigure('box');
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
        
            % save
            obj.saveFigure('board');
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
            
            % save
            obj.saveFigure('costs');
            
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
                    scatter(cx, cy, ...
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
        function [train, val, test] = error(obj, d)
            % error for `train`, `validation` and `test` datasets
            train = getError(...
                obj.X(obj.dataIndexes.train), ...
                obj.Y(obj.dataIndexes.train) ...
            );
            val = getError(...
                obj.X(obj.dataIndexes.val), ...
                obj.Y(obj.dataIndexes.val) ...
            );
            test = getError(...
                obj.X(obj.dataIndexes.test), ...
                obj.Y(obj.dataIndexes.test) ...
            );
            
            % Local Functions
            function err = getError(x, y)
                % number of epochs
                n = obj.numOfEpochs;
                
                % error
                err = zeros(n, 1);
                for epoch = 1:n
                    obj.dag.load_epoch(epoch);
                    y_ = obj.dag.out(x);
                    
                    err(epoch) = mean(...
                        arrayfun(...
                            @(i) d(y{i}, y_{i}), ...
                            1:length(x)...
                        )...
                    );
                
                    % NaN -> 0
                    if isnan(err(epoch))
                        err(epoch) = 0;
                    end
                end
            end
        end
        function plotErrors(obj, d)
            % Properties
            lineWidth = 2;
            yScale = 'log';
            
            [train, val, test] = obj.error(d);
            epochs = 0:(length(train) - 1);
            
            plotTrainValTestErrors();
            setAxes();
            
            % save
            obj.saveFigure('errors');
            
            % Local Functions
            function plotTrainValTestErrors()
                % figure
                Viz.figure('CNN - Errors [Training, Validation, Test]');
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
                        'XTick', unique([0, epochs(end)]) ...
                    );
                    % - y
%                     set(gca, ...
%                         'YTick', ...
%                         unique(sort([...
%                             0, ...
%                             max([train, val, test]) ...
%                         ])) ...
%                     );
                end
                function setLabels()
                    % labels
                    xlabel('Epoch');
                    ylabel('Error');
                end
                function setTitle()
                    % title
                    title(func2str(d));
                end
                function setLegend()
                    % legend
                    legend('Training', 'Validation', 'Test');
                end
            end
        end
        function plotCorr(obj)
            % Properties
            lineWidth = 2;
            yScale = 'linear';
            
            train = obj.corr.train;
            val = obj.corr.val;
            test = obj.corr.test;
            epochs = 0:(length(train) - 1);
            
            plotTrainValTestErrors();
            setAxes();
            
            % save
            obj.saveFigure('corr');
            
            % Local Functions
            function plotTrainValTestErrors()
                % figure
                Viz.figure('CNN - Errors [Training, Validation, Test]');
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
                        'XTick', unique([0, epochs(end)]) ...
                    );
                    % - y
%                     set(gca, ...
%                         'YTick', ...
%                         unique(sort([...
%                             0, ...
%                             max([train, val, test]) ...
%                         ])) ...
%                     );
                end
                function setLabels()
                    % labels
                    xlabel('Epoch');
                    % ylabel('Corr');
                end
                function setTitle()
                    % title
                    title('Linear Correlation');
                end
                function setLegend()
                    % legend
                    legend('Training', 'Validation', 'Test');
                end
            end
        end
    end
    
    % Plot Bias
    methods
        function plotBias(obj, biasName)
            
            % plot for all filters
            if ~exist('biasName', 'var')
                plotBiasAll();

                return;
            end
            
            % parameters
            lineWidth = 1.5;
            markerSize = 20;
            markerColor = [0.8500, 0.3250, 0.0980];
            hlineColor = [0.4660, 0.6740, 0.1880];
            
            bias = obj.params.(biasName);
            
            y = [bias.history{:}];
            x = 0 : (length(y) - 1);
            
            xMin = min(x);
            xMax = max(x);
            yMin = min(y);
            yMax = max(y);
            xBest = bias.minValCost.index - 1;
            yBest = y(bias.minValCost.index);
            
            if xMax == xMin
                xMax = xMax + eps;
            end
            if yMax == yMin
                yMax = yMax + eps;
            end
            
            % figure
            Viz.figure(sprintf('Filter: %s', biasName));
            % plot history
            plot(x, y, 'LineWidth', lineWidth);
            title(biasName);
            xlabel('Epoch');
            hold('on');
            % expected value (horizontal line)
            drawHorizontalLine(bias.expected);
            % plot circle
            drawCircle(xBest, yBest);
            hold('off');
            % axis
            setAxes();
            
            % save
            obj.saveFigure(biasName);

            % Local Functions
            function plotBiasAll()
                for i = 1:length(obj.biasNames)
                    biasName = obj.biasNames{i};
                    obj.plotBias(biasName);
                end
            end
            function drawCircle(cx, cy)
                plot(...
                    cx, ...
                    cy, ...
                    'Color', markerColor, ...
                    'Marker', '.', ...
                    'MarkerSize', markerSize ...
                );
            end
            function drawHorizontalLine(y)
                ax = gca;
                line(...
                    ax.XLim, ...
                    [y, y], ...
                    'Color', hlineColor, ...
                    'LineStyle', ':', ...
                    'LineWidth', lineWidth ...
                );
            end
            function setAxes()
                % grid
                grid('on');
                box('off');

                % ticks
                % - x
                %todo: writ `threeticks` method
                set(gca, ...
                    'XTick', ...
                    unique([xMin, xBest, xMax]) ...
                );
                xlim([xMin, xMax]);
                % - y
                set(gca, ...
                    'YTick', ...
                    unique([yMin, yBest, yMax]) ...
                );
                ylim([yMin, yMax]);
            end
        end
    end
    
    % Plot Filters
    methods
        function playFilterVideo(obj, filterName)
            % Play or save video of learning process of the specific filter
            
            % play all filters
            if ~exist('filterName', 'var')
                playFilterVideoAll();
                
                return;
            end
            
            filterVideoFilename = obj.getFilterVideoFilename(filterName);
            if ~exist(filterVideoFilename, 'file')
                obj.saveFilterVideo(filterName, filterVideoFilename);
            end
            
            Viz.playVideo(filterVideoFilename);
            fprintf('Video path: ''%s\''', filterVideoFilename);
            
            % Local Functions
            function playFilterVideoAll()
                for i = 1:length(obj.filterNames)
                    filterName = obj.paramNames{i};
                    obj.playFilterVideo(filterName);
                end
            end
            
        end
        function filterVideoFilename = getFilterVideoFilename(obj, filterName)
            filterVideoFilename = fullfile(...
                obj.path, ...
                Path.VIDEOS_DIR, ...
                [filterName, '.mp4'] ...
            );
        end
        function plotFilterOldOld(obj, filterName)
            
            % plot for all filters
            if ~exist('filterName', 'var')
                plotFilterAll();

                return;
            end
            
            % Parameters
            lineWidth = 2;
            filterExpectedColor = [0.4660, 0.6740, 0.1880];
            filterInitialColor = [0.8500, 0.3250, 0.0980];
            filterMinErrorColor = [0.9290, 0.6940, 0.1250];
            limits = obj.getFilterLimits(filterName);
            
            Viz.figure(sprintf('Filter: %s', filterName));
            
            plotFilterExpected();
            hold('on');
            plotFilterInitial();
            plotFilterMinError();
            setAxes()
            setTitle();
            
            % save
            obj.saveFigure(filterName);
            
            % Local Functions
            function plotFilterAll()
                for i = 1:length(obj.filterNames)
                    filterName = obj.filterNames{i};
                    obj.plotFilter(filterName);
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
            function setTitle()
                title(sprintf(...
                    'Filter %s - Minimum Validation Error is %g in Epoch #%d', ...
                    filterName, ...
                    obj.params.(filterName).minValCost.value, ...
                    obj.params.(filterName).minValCost.index - 1 ...
                ));
            end
        end
        function plotFilterOld(obj, filterName)
            
            % plot for all filters
            if ~exist('filterName', 'var')
                plotFilterAll();

                return;
            end
            
            % Parameters
            lineWidth = 2;
            
            colors = makeColors(5);
            
            filterInitialColor = colors(1, :);
            filterEpochOneThirdColor = colors(2, :);
            filterEpochTwoThirdColor = colors(3, :);
            filterMinErrorColor = colors(4, :);
            filterExpectedColor = colors(5, :);
            
            limits = obj.getFilterLimits(filterName);
            
            [~, index] = min(obj.costs.val);
            
            Viz.figure(sprintf('Filter: %s', filterName));
            
            plotFilterInitial();
            hold('on');
            plotFilterEpochOneThird();
            plotFilterEpochTwoThird();
            plotFilterMinError();
            plotFilterExpected();
            hold('off');
            
            setAxes()
            setTitle();
            
            % save
            obj.saveFigure(filterName);
            
            % Local Functions
            function plotFilterAll()
                for i = 1:length(obj.filterNames)
                    filterName = obj.filterNames{i};
                    obj.plotFilter(filterName);
                end
            end
            function plotFilterInitial()
                plot(obj.params.(filterName).initial, ...
                    'DisplayName', 'Initial value', ...
                    'LineWidth', lineWidth, ...
                    'Color', filterInitialColor ...
                );
            end
            function plotFilterEpochOneThird()
                oneThird = ceil(index * (1/3));
                plot(obj.params.(filterName).history{oneThird}, ...
                    'DisplayName', sprintf('One Third (#%d)', oneThird - 1), ...
                    'LineWidth', lineWidth, ...
                    'Color', filterEpochOneThirdColor ...
                );
            end
            function plotFilterEpochTwoThird()
                twoThird = ceil(index * (2/3));
                plot(obj.params.(filterName).history{twoThird}, ...
                    'DisplayName', sprintf('Two Third (#%d)', twoThird - 1), ...
                    'LineWidth', lineWidth, ...
                    'Color', filterEpochTwoThirdColor ...
                );
            end
            function plotFilterMinError()
                plot(obj.params.(filterName).history{index}, ...
                    'DisplayName', sprintf('Min Val Error (#%d)', index - 1), ...
                    'LineWidth', lineWidth, ...
                    'Color', filterMinErrorColor ...
                );
            end
            function plotFilterExpected()
                plot(obj.params.(filterName).expected, ...
                    'DisplayName', 'Expected', ...
                    'LineWidth', lineWidth, ...
                    'Color', filterExpectedColor ...
                );
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
            function setTitle()
                title(sprintf(...
                    'Filter %s - Minimum Validation Error is %g in Epoch #%d', ...
                    filterName, ...
                    obj.params.(filterName).minValCost.value, ...
                    obj.params.(filterName).minValCost.index - 1 ...
                ));
            end
            function colors = makeColors(n)
                colors = zeros(n, 3);
                
                map = summer;
                m = length(map);
                
                for i = 1:n
                    j = ceil((i / n) * m);
                    colors(i, :) = map(j, :);
                end
            end
        end
        function plotFilter(obj, filterName)
            
            % plot for all filters
            if ~exist('filterName', 'var')
                plotFilterAll();

                return;
            end
            
            % Parameters
            lineWidth = 2;
            filterExpectedColor = [0.4286, 0.4286, 0.4286];
            filterInitialColor = [0.9290, 0.6940, 0.1250];
            filterMinErrorColor = [0.4940, 0.1840, 0.5560];
            
            Viz.figure(sprintf('Filter: %s', filterName));
            
            plotFilterExpected();
            hold('on');
            plotFilterInitial();
            plotFilterMinError();
            setAxes()
            setTitle();
            
            % save
            obj.saveFigure([filterName, '-new']);
            
            % Local Functions
            function plotFilterAll()
                for i = 1:length(obj.filterNames)
                    filterName = obj.filterNames{i};
                    obj.plotFilter(filterName);
                end
            end
            function plotFilterExpected()
                plot(flip(obj.params.(filterName).expected), ...
                    'DisplayName', 'Ground truth', ...
                    'LineWidth', lineWidth, ...
                    'Color', filterExpectedColor ...
                );
            end
            function plotFilterInitial()
                plot(flip(obj.params.(filterName).initial), ...
                    'DisplayName', 'Initial value', ...
                    'LineWidth', lineWidth, ...
                    'Color', filterInitialColor ...
                );
            end
            function plotFilterMinError()
                [~, index] = min(obj.costs.val);
                plot(flip(obj.params.(filterName).history{index}), ...
                    'DisplayName', sprintf('Min Val Error (#%d)', index - 1), ...
                    'LineWidth', lineWidth, ...
                    'Color', filterMinErrorColor ...
                );
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
                    axis('tight');
                    set(gca, 'XTick', []);
                    set(gca, 'YTick', []);
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
                    lgd.Location = 'northeast';
                    lgd.Box = 'off';
                end
            end
            function setTitle()
                title(sprintf(...
                    'Filter %s - Minimum Validation Error is %g in Epoch #%d', ...
                    filterName, ...
                    obj.params.(filterName).minValCost.value, ...
                    obj.params.(filterName).minValCost.index - 1 ...
                ));
            end
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
            
            if xMax == xMin
                xMax = xMax + eps;
            end
            if yMax == yMin
                yMax = yMax + eps;
            end
            limits = [xMin, xMax, yMin, yMax];
        end
        
        function plotFilterHistory(obj, filterName, epochs)
            % Plot evolution history of a filter during the learning 
            % process in a grid
            %
            % Parameters
            % ----------
            % - filterName: char vector
            %   Name of the filter
            
            % plot for all filters
            if ~exist('filterName', 'var')
                plotFilterHistoryForAllFilters();
                
                return;
            end
            
            numberOfEpochs = length(obj.params.(filterName).history());
            
            % plot each `n` epochs in a new figure
            if ~exist('epochs', 'var')
                plotFilterHistoryForAllEpochs();
                
                return;
            end
            
            % properties
            fontsize = 6;
            expectedColor = Viz.VAL_COLOR;
            expectedLineWidth = 1.5;
            minValCostColor = Viz.TEST_COLOR;
            minValCostLineWidth = 1.5;
            
            % default values
            if ~exist('epochs', 'var')
                epochs = 1:numberOfEpochs;
            end
            
            % history
            x = obj.params.(filterName).history();
            
            % number of samples
            numberOfSamples = length(epochs);
            
            % subplot grid
            [cols, rows] = getColsRows(numberOfSamples);
            
            % figure
            Viz.figure(sprintf('Filter History - %s', filterName));
            
            % plot
            limits = obj.getFilterLimits(filterName);
            hWaitbar = waitbar(0, 'Plot Filter History...');
            
            minValCostIndex = obj.params.(filterName).minValCost.index;
            for sampleIndex = 1 : numberOfSamples
                epoch = epochs(sampleIndex);
                % sample
                subplot(rows, cols, sampleIndex);
                h = plot(x{epoch});
                
                % expected
                if epoch == 1
                    hold('on');
                    plot(obj.params.(filterName).expected, ...
                        'Color', expectedColor, ...
                        'LineWidth', expectedLineWidth);
                    hold('off');
                end
                
                % red sample
                if epoch == minValCostIndex
                   set(h, ...
                       'Color', minValCostColor, ...
                       'LineWidth', minValCostLineWidth ...
                   ); 
                end
                
                setAxis();
                showTitle(epoch);
                
                waitbar(sampleIndex / numberOfSamples)
            end
            close(hWaitbar);
            
            % super-title
            suptitle(...
                sprintf(...
                    'Learning History of Parameter %s', ...
                    filterName ...
                ) ...
            );
            
            % Local Functions
            function plotFilterHistoryForAllFilters()
                for i = 1:length(obj.paramNames)
                    paramName = obj.paramNames{i};
                    
                    % filter names start with `w`
                    if paramName(1) == 'w'
                        obj.plotFilterHistory(paramName);
                    end
                end
            end
            function plotFilterHistoryForAllEpochs()
                n = 100; % number of epochs in each figure
                s = 1; % stard epoch
                f = n; % finish epoch
                
                while s < numberOfEpochs
                    if f > numberOfEpochs
                        f = numberOfEpochs;
                    end
                    
                    obj.plotFilterHistory(filterName, s:f);
                    
                    s = s + n;
                    f = f + n;
                end
            end
            function [cols, rows] = getColsRows(n)
                % cols > rows
                % - cols
                cols = ceil(sqrt(n));
                % - rows
                rows = ceil(n / cols);
            end
            function setAxis()
                axis(limits);
                
                ax = gca;
                ax.XAxisLocation = 'origin';
                ax.YAxis.Visible = 'off'; 
                
                Viz.hideticks();
            end
            function showTitle(i)
                title(num2str(i - 1));
                set(gca, 'FontSize', fontsize);
            end
        end
    end
    
    % Plot Parameters
    methods
        function plotParameters(obj)
            obj.plotBias();
            obj.plotFilter();
        end
    end
    
    % Plot Expected/Actual Responses
    methods
        function plotExpectedActualOutputs(obj, epoch, indexes)
            % Plot epoch/ground-truth in a grid pattern
            %
            % Parameters
            % ----------
            % - epoch: number
            %   Target epoch
            % - indexes: number[]
            %   Index of desired samples
            
            % plot for epoch with min validation cost
            if ~exist('epoch', 'var')
                [~, epoch] = min(obj.costs.val);
            end
            
            % plot for all filters
            if ~exist('indexes', 'var')
                % indexes = 1:obj.N;
                indexes = 1:min(100, obj.N);
            end
            
            if length(indexes) > 100
                obj.plotExpectedActualOutputs(epoch, indexes(1:100));
                obj.plotExpectedActualOutputs(epoch, indexes(101:end));
                return;
            end
            
            % actual response
            obj.dag.load_epoch(epoch);
            Y_ = obj.dag.out(obj.X);
            
            % properties
            fontsize = 6;
            expectedColor = Viz.VAL_COLOR;
            expectedLineWidth = 1.5;
            minValCostColor = Viz.TEST_COLOR;
            minValCostLineWidth = 1.5;
            
            % number of samples
            numberOfSamples = length(indexes);
            
            % subplot grid
            [cols, rows] = getColsRows(numberOfSamples);
            
            % figure
            Viz.figure(sprintf('Expected vs. Actual Responses for Epoch #%d', epoch - 1));
            
            % plot
            limits = getLimits();
            hWaitbar = waitbar(0, 'Plot Expected/Actual Responses...');
            
            for index = 1 : numberOfSamples
                sampleIndex = indexes(index);
                % sample
                subplot(rows, cols, index);
                plot(obj.Y{sampleIndex});
                hold('on');
                plot(Y_{sampleIndex});
                hold('off');
                
                setAxis();
                showTitle(sampleIndex);
                
                waitbar(index / numberOfSamples)
            end
            close(hWaitbar);
            
            % super-title
            suptitle(sprintf('Expected vs. Actual Responses for Epoch #%d', epoch - 1));
            
            % save
            obj.saveFigure('prediction');
            
            % Local Functions
            function limits = getLimits()
                xMin = 1;
                xMax = length(obj.Y{1});
                yMin = min(min(cellfun(@min, obj.Y)), min(cellfun(@min, Y_)));
                yMax = max(max(cellfun(@max, obj.Y)), max(cellfun(@max, Y_)));
                limits = [xMin, xMax, yMin, yMax];
            end
            function [cols, rows] = getColsRows(n)
                % cols > rows
                % - cols
                cols = ceil(sqrt(n));
                % - rows
                rows = ceil(n / cols);
            end
            function setAxis()
                try
                    axis(limits);
                catch
                end
                
                ax = gca;
                ax.XAxisLocation = 'origin';
                ax.YAxis.Visible = 'off'; 
                
                Viz.hideticks();
            end
            function showTitle(i)
                title(num2str(i));
                set(gca, 'FontSize', fontsize);
            end
        end
        function plotExpectedActualOutputsOverall(obj, epoch)
            % Plot expected/actual responses for a given epoch
            %
            % Parameters
            % ----------
            % - epoch: number
            %   Target epoch
            
            % plot for epoch with min validation cost
            if ~exist('epoch', 'var')
                [~, epoch] = min(obj.costs.val);
            end
            
            % actual response
            obj.dag.load_epoch(epoch);
            Y_ = obj.dag.out(obj.X);
            
            % train, val, test indexes
            idx = cell(3, 1);
            idx{1} = obj.dataIndexes.train;
            idx{2} = obj.dataIndexes.val;
            idx{3} = obj.dataIndexes.test;
            
            % properties
            titles = {'Training', 'Validation', 'Test'};
            names = {'train', 'val', 'test'};
            expectedColor = 'black';
            colors = [Viz.TRAIN_COLOR; Viz.VAL_COLOR; Viz.TEST_COLOR];
            
            % figure
            Viz.figure(sprintf('Expected vs. Actual Responses for Epoch #%d', epoch - 1));
            
            % plot
            for i = 1:3
                y = concat(obj.Y(idx{i}));
                y_ = concat(Y_(idx{i}));
                
                subplot(3, 1, i);
                
                plot(y, ...
                    'DisplayName', 'Expected', ...
                    'Color', expectedColor ...
                );
                hold('on');
                
                plot(y_, ...
                    'DisplayName', 'Actual', ...
                    'Color', colors(i, :) ...
                );
                hold('off');
                
                legend('show');
                title(...
                    sprintf('%s (mse: %g, corr: %g)', ...
                        titles{i}, ...
                        obj.costs.(names{i})(epoch), ...
                        obj.corr.(names{i})(epoch) ...
                    ) ...
                );
            end
            
            % super-title
            suptitle(sprintf('Expected vs. Actual Responses for Epoch #%d', epoch - 1));
            
            % save
            obj.saveFigure('prediction-overall');
            
            % Local Functions
            function X = concat(x)
                X = vertcat(x{:});
            end
        end
    end
        
    % Utils
    methods
        function config = readConfig(obj)
            config = jsondecode(fileread(...
                fullfile(obj.path, obj.CONFIG_FILENAME)...
            ));
        end
        function saveFigure(obj, filename)
            % Save curret figure to file `filename`
            %
            % Parameters
            % ----------
            % - filename: char vector
            %   Path of saved figure
            
            saveas(...
                gcf, ...
                fullfile(obj.path, Path.FIGURES_DIR, filename), ...
                Viz.FORMATTYPE ...
            );
        end
        function saveData(obj)
            % go to best epoch
            [~, epoch] = min(obj.costs.val);
            obj.dag.load_epoch(epoch);
            Y_ = obj.dag.out(obj.X);
            
            % train, val, test indexes
            idxTrain = obj.dataIndexes.train;
            idxVal = obj.dataIndexes.val;
            idxTest = obj.dataIndexes.test;
            
            % data
            data = struct();
            
            % indexes (train, val, test)
            data.idx.train = idxTrain;
            data.idx.val = idxVal;
            data.idx.test = idxTest;
            
            
            % - all
            data.x = obj.X;
            data.y = obj.Y;
            data.y_ = Y_;
            
            data.X = concat(data.x);
            data.Y = concat(data.y);
            data.Y_ = concat(data.y_);
            
            % - train
            data.train.x = obj.X(idxTrain);
            data.train.y = obj.Y(idxTrain);
            data.train.y_ = Y_(idxTrain);
            
            data.train.X = concat(data.train.x);
            data.train.Y = concat(data.train.y);
            data.train.Y_ = concat(data.train.y_);
            
            % - val
            data.val.x = obj.X(idxVal);
            data.val.y = obj.Y(idxVal);
            data.val.y_ = Y_(idxVal);
            
            data.val.X = concat(data.val.x);
            data.val.Y = concat(data.val.y);
            data.val.Y_ = concat(data.val.y_);
            
            % - test
            data.test.x = obj.X(idxTest);
            data.test.y = obj.Y(idxTest);
            data.test.y_ = Y_(idxTest);
            
            data.test.X = concat(data.test.x);
            data.test.Y = concat(data.test.y);
            data.test.Y_ = concat(data.test.y_);
            
            % params
            % - all
            data.params = obj.params;
            % - best
            names = fieldnames(obj.params);
            for i = 1:length(names)
                name = names{i};
                data.(name) = obj.params.(name).history{epoch};
            end
            
            % errors
            % - mse
            data.mse = obj.costs;
            % - corr
            data.corr = obj.corr;
            
            % save
            filename = fullfile(obj.path, 'overall');
            save(filename, '-struct', 'data');
            
            % Local Functions
            function X = concat(x)
                X = vertcat(x{:});
            end
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
