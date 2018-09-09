classdef DataUtils < handle
    %DATAUTILS contains data-utilities functions
    
    
    methods (Static)
        function output = divide_to_subvectors(input, l, d)
            % Splits input vector to subvectors
            %
            %   Parameters
            %   ----------
            %   - input : double vector
            %       input vector
            %   - l : int
            %       length of sub-vector
            %   - d : int
            %       delta between two sub-vectors
            %
            %   Returns
            %   -------
            %   - output : 2d double array
            %       array of splited sub-vectors(each row).
            %
            %   Examples
            %   --------
            %   1.
            %       >>> input = [1, 2, 3, 4, 5];
            %       >>> l = 3;
            %       >>> d = 2;
            %       >>> divide_to_subvectors(input, l, d)
            %       [
            %           [1, 2, 3];
            %           [3, 4, 5]
            %       ]
            
            % let 'm' is the number of splited sub-vectors and 'n' is the
            % length of 'input', so
            % (m - 1)d + l = n
            % m = floor((n - l) / d) + 1
            n = length(input);
            m = floor((n - l) / d) + 1;
            output = zeros(m, l);
            
            for i = 1 : m
                begin_index = (i - 1) * d + 1;
                end_index = begin_index + l - 1;
                
                output(i, :) = input(begin_index : end_index);
            end
            
%             output = [];
%             i = 1;
%             while (i + l - 1) <= length(input)
%                 output(end + 1, :) = input(i : i + l - 1);
%                 
%                 i = i + d;
%             end
        end
        function output = divide_timeseries(input, dt_sec, l_sec, d_sec)
            %DIVIDE_TIMESERIES splits input signal to sub-signals
            %
            %   Parameters
            %   ----------
            %   - input : double vector
            %       input vector
            %   - dt_sec : double
            %       time resolution in seconds
            %   - l_sec : double
            %       length of sub-signal in seconds
            %   - d_sec : double
            %       delta between two sub-signlas in secondss
            %
            %   Returns
            %   -------
            %   - output : 2d double array
            %       array of splited sub-vectors(each row)
            %
            %   Examples
            %   --------
            %   1.
            %       >>> input = [1, 2, 3, 4, 5];
            %       >>> dt_sec = 0.001;
            %       >>> l_sec = 0.002;
            %       >>> d_sec = 0.001;
            %       >>> divide_timeseries(input, dt_sec, l_sec, d_sec)
            %       [
            %           [1, 2],
            %           [2, 3],
            %           [3, 4],
            %           [4, 5]
            %       ]
            
            % default values
            switch nargin
                case 1
                    dt_sec = 0.001;
                    l_sec = 0.001;
                    d_sec = l_sec;
                case 2
                    l_sec = 0.001;
                    d_sec = l_sec;
                case 3
                    d_sec = l_sec;
            end
            
            l = floor(l_sec / dt_sec);
            d = floor(d_sec / dt_sec);
            output = DataUtils.divide_to_subvectors(input, l, d);
        end
        function output = divide_to_samelength_subvectors(input, m)
            %DIVIDE_TO_SAMELENGTH_SUBVECTORS splits input vector to same-legnth subvectors
            %
            %   Parameters
            %   ----------
            %   - input : double vector
            %       input vector
            %   - m : int
            %       number of sub-vectors
            %
            %   Returns
            %   -------
            %   - output : 2d double array
            %       array of splited sub-vectors(each row)
            %
            %   Examples
            %   --------
            %   1.
            %       >>> input = [1, 2, 3, 4, 5];
            %       >>> m = 2;
            %       >>> divide_to_samelength_subvectors(input, l, d)
            %       [
            %           [1, 2],
            %           [3, 4]
            %       ]
            
            % length of each sub-vector
            l = floor(length(input) / m);
            % remove residual elements(mod(lenght(input), m) == 0)
            input = input(1 : m * l);
            % divide
            output = divide_to_subvectors(input, l, l);
        end
        function output = downsample_vector(input, m)
            %DOWNSAMPLE_VECTOR donwsample 'input' to 'output' with length 'm'
            %
            %   Parameters
            %   ----------
            %   - input : double vector
            %       input vector
            %   - m : int
            %       length of output
            %
            %   Returns
            %   -------
            %   - output : double vector
            %       downsampled vector
            %
            %   Examples
            %   1.
            %       >>> input = [1, 2, 3, 4, 5];
            %       >>> m = 3;
            %       >>> downsample_vector(input, m)
            %       [1, 3, 5]
            
            % let 'n' is a length of 'input' and 'd' is the sampling rate, so
            % (m - 1)d + 1 = n
            % d = floor((n - 1) / (m - 1))
            
            n = length(input);
            d = floor((n - 1) / (m - 1));
            % remove residual elements(mod(n, m) == 0)
            n = m * d;
            input = input(1 : n);
            output = input(1 : d : n);
        end
        function output = resize(input, output_size, method)
            %RESIZE resizes 'input' to 'output' with length 'm'
            %
            %   Parameters
            %   ----------
            %   - input : double array
            %       input vector
            %   - output_size : int array
            %       size of output
            %   - method : char vector (default is 'bicubic')
            %       method of interpolation such as 'nearest', 'bilinear'
            %       and 'bicubic'
            %
            %   Returns
            %   -------
            %   - output : double array
            %       resized vector
            %
            %   Examples
            %   1.
            %       >>> input = [1, 2, 3, 4];
            %       >>> output_size = [2, 1]
            %       >>> method = 'bilinear';
            %       >>> resize(input, output_size, method)
            %       [1.5, 3.5]
            
            % default values
            if nargin < 3
                method = 'bicubic';
            end
            
            output = imresize(...
                input, output_size, ...
                method, ...
                'Antialiasing', false);
        end
    end
    
    % Make Random Data
    methods (Static)
        function make_data(n, l, filename, generator)
            % Make random `data` file
            %
            % Parameters
            % ----------
            % - n : int
            %   number of samples
            % - l : int
            %   length of each sample
            % - filename: char vector
            %   filename of saved file
            % - generator : handle function (default is @randn)
            %   generator function such as `randn`, `rand`, ...
            
            % default generator
            if ~exist('generator', 'var')
                generator = @randn;
            end
            
            % db
            data = struct();
            data.x = cell(n, 1);
            data.y = cell(n, 1);
            
            % - x, y
            for i = 1:n
                data.x{i} = generator([l, 1]);
                data.y{i} = generator([l, 1]);
            end
            
            % - save
            save(fullfile(Path.DATA_DIR, filename), '-struct', 'data');
            clear('data');
        end
        function transformAndSaveRealData(inFilename)
            
            % parameters
            % - output directory
            outDirData = './assets/data';
            outDirParams = './assets/ground-truth';
%             inputField = 'input_PSTHsmooth';
%             outputField = 'output_PSTHmoresmooth';
%             inputField = 'input_spksmooth';
%             outputField = 'output_spkmoresmooth';
            inputField = 'input_ce';
            outputField = 'output_ce';
            
            % - length of input sample
            li = 50;
            % li = 2000;
            % - length of output sample
            lo = 26;
            % lo = 1976;
            % - offset between two consecutive data
            d = 5;
            % d = li;
            % - begin index of data
            begin = 1;
            % - scale of output
            scale = 0.1;
            scale = 1;
            % - max(abs(sample)) must be greater thatn threshold
            th = 0.1;
            th = 0.01;
            th = 0;

            % make data
            data = load(inFilename);
            
            % divide input
            input = data.(inputField)(begin:end);
            % - make zero mean, unit variance
            input = (input - mean(input)) / std(input);
            input = DataUtils.divide_to_subvectors(input, li, d);
            
            x = num2cell(input', 1)';
            
            % divide output
            output = data.(outputField)(begin:end);
            output = DataUtils.divide_to_subvectors(output, li, d) * scale;
            y = num2cell(output', 1)';
            
            y = cellfun(@(s) s(1:lo), y, 'UniformOutput', false);
            
            % filter x, y
            v = cellfun(@(s) max(abs(s)), y);
            i = find(v > th);
            x = x(i);
            y = y(i);
            
            
            % save db
            [~, name, ~] = fileparts(inFilename);
            outFilename = fullfile(outDirData, name);
            save(outFilename, 'x', 'y');
            
            % make parameters
            % b_A
            b_A = 0;
            % b_B
            b_B = 0;
            % b_G
            b_G = 0;
            
            % w_A
            w_A = data.FA_ds(:);
            w_A = w_A(end:-1:1);
            % w_B
            w_B = data.FB_ds(:);
            w_B = w_B(end:-1:1);
            % w_G
            w_G = 1;
            
            % - save
            outFilename = fullfile(outDirParams, name);
            save(outFilename, 'b_A', 'b_B', 'b_G', 'w_A', 'w_B', 'w_G');
        end
        function B = model1(w_B, b_B)
            % B
            % - linear
            BL = @(x) conv(x, w_B, 'valid') + b_B;
            % - nonlinear
            BN = @(x) max(0, x);
            % - LN
            B = @(x) BN(BL(x));
        end
        function G = model2(w_B, b_B, w_A, w_G, b_G)
            % B
            % - linear
            BL = @(x) conv(x, w_B, 'valid') + b_B;
            % - nonlinear
            BN = @(x) max(0, x);
            % - LN
            B = @(x) BN(BL(x));
            
            % A
            % - linear
            AL = @(x) conv(x, w_A, 'valid');
            % - nonlinear
            AN = @(x) x;
            % - LN
            A = @(x) AN(AL(x));
            
            % G
            % - linear
            GL = @(x) w_G * (B(x) - A(x)) + b_G;
            % - nonlinear
            GN = @(x) max(0, x);
            % - LN
            G = @(x) GN(GL(x)); 
        end
        function G = model3(w_B, w_A, b_A, w_G, b_G)
            % B
            % - linear
            BL = @(x) conv(x, w_B, 'valid');
            % - nonlinear
            BN = @(x) x;
            % - LN
            B = @(x) BN(BL(x));
            
            % A
            % - linear
            AL = @(x) conv(x, w_A, 'valid') + b_A;
            % - nonlinear
            AN = @(x) logsig(x);
            % - LN
            A = @(x) AN(AL(x));
            
            % G
            % - linear
            GL = @(x) w_G * (B(x) .* A(x)) + b_G;
            % - nonlinear
            GN = @(x) max(0, x);
            % - LN
            G = @(x) GN(GL(x));
        end
        function G = model4(w_B, b_B, w_A, b_A, w_G, b_G)
            % B
            % - linear
            BL = @(x) conv(x, w_B, 'valid') + b_B;
            % - nonlinear
            BN = @(x) max(0, x);
            % - LN
            B = @(x) BN(BL(x));
            
            % A
            % - linear
            AL = @(x) conv(x, w_A, 'valid') + b_A;
            % - nonlinear
            AN = @(x) logsig(x);
            % - LN
            A = @(x) AN(AL(x));
            
            % G
            % - linear
            GL = @(x) w_G * (B(x) .* A(x)) + b_G;
            % - nonlinear
            GN = @(x) max(0, x);
            % - LN
            G = @(x) GN(GL(x)); 
        end
        function makeAndSaveExpectedOutputs(dataFilename, paramsFilename, outFilename)
            % data
            data = load(dataFilename);
            x = data.x;
            y = data.y;
            
            % parameters
            params = load(paramsFilename);
            w_B = params.w_B;
            b_B = params.b_B;
            w_A = params.w_A;
            b_A = params.b_A;
            w_G = params.w_G;
            b_G = params.b_G;
            
            % models
            f1 = DataUtils.model1(w_B, b_B);
            f2 = DataUtils.model2(w_B, b_B, w_A, w_G, b_G);
            f3 = DataUtils.model3(w_B, w_A, b_A, w_G, b_G);
            f4 = DataUtils.model4(w_B, b_B, w_A, b_A, w_G, b_G);
            
            % expected outputs
            y1 = out(f1);
            y2 = out(f2);
            y3 = out(f3);
            y4 = out(f4);
            
            % save
            save(outFilename, 'x', 'y', 'y1', 'y2', 'y3', 'y4');
            
            % Local Functions
            function y = out(f)
                y = cellfun(f, x, 'UniformOutput', false);
            end
        end
    end
    
    % Make Random Parameters
    methods (Static)
        function make_params(l, filename)
            % Make random `data` file
            %
            % Parameters
            % ----------
            % - l : int vector
            %   length of each sample
            % - filename: char vector
            %   filename of saved file
            
            params = struct();
            % b_A
            params.b_A = 0;
            % b_B
            params.b_B = 0;
            % b_G
            params.b_G = 0;
            
            x = linspace(0, 2 * pi, l)';
            % w_A
            params.w_A = -sin(x);
            % w_B
            params.w_B = sin(x);
            % w_G
            params.w_G = cos(x);
            
            % - save
            save(fullfile(Path.GROUND_TRUTH_DIR, filename), '-struct', 'params');
            clear('params');
        end
    end
    
    % Error
    methods (Static)
        function e = error(x, y, f, d)
            % Compute averaged error of a model
            
            % Parameters
            % ----------
            % - x: number[][]
            %   Input
            % - y: number[][]
            %   Output
            % - f: (x: number[]) => number[]
            %   Model
            % - d: (y: number[], y_: number[]) => number
            %   Distance
            
            if ~exist('d', 'var')
                d = @(u, v) norm(u - v);
            end
            
            e = mean(...
                arrayfun(...
                    @(i) d(y{i}, f(x{i})), ...
                    1:length(x)...
                )...
            );
        end
    end
    
    methods (Static)
        function kernel = make_gaussian_kernel( n, sigma )
            %MAKE_GAUSSIAN_KERNEL makes 1d gaussian kernel
            %
            %   Parameters
            %   ----------
            %   - n : int
            %       length of output kernel is 2*n+1
            %   - sigma : double
            %       std of gaussian kernel
            %
            %   Returns
            %   -------
            %   - kernel : double array
            %       sum of kernel must be 1
            
            x = -n:n;
            kernel = exp(-((x ./ sigma) .^ 2));
            kernel = kernel ./ sum(kernel);
        end
        function psth = spks_to_psth( spks, window_size_sec, kernel_size_sec, dt_sec )
            %SPKS_TO_PSTH make psth (peri-stimulus time histogram) from input spike trains
            %
            %   Parameters
            %   ----------
            %   - spks : 2d double array
            %       spikes
            %   - window_size_sec : double (default is 0.001)
            %       window of computing psth. unit is second.
            %   - kernel_size_sec : double (default is 0.001)
            %       size of kernel of gaussian smoothing. unit is second.
            %   - dt_sec : double (default is 0.001)
            %       time resolution. unit is second;
            %
            %   Examples
            %   --------
            %   1.
            %       >>> spks = [[1, 2]; [3, 4]];
            %       >>> spks_to_psth(spks)
            %       [2000, 3000]
            
            % default values
            switch nargin
                case 1
                    window_size_sec = 0.001;
                    kernel_size_sec = 0.001;
                    dt_sec = 0.001;
                case 2
                    kernel_size_sec = 0.001;
                    dt_sec = 0.001;
                case 3
                    dt_sec = 0.001;
            end
            
            % number of trials
            m = size(spks, 1);
            % length of each trial
            n = size(spks, 2);
            
            % number of samples for computing firing rate in each trial
            window_size = floor(window_size_sec / dt_sec);
            % window_size_sec = window_size * dt_sec;
            
            % mod(n, window-size) must be equals 0
            residual = mod(n, window_size);
            if residual ~= 0
                spks = [spks, zeros(m, window_size - residual)];
            end
            
            % number of elements in each window in all trials
            batch_size = m * window_size;
            batch_size_sec = batch_size * dt_sec;
            
            % make psth
            psth = zeros(1, n);
            % - i is index of psth
            i = 1;
            % - j is index of spks
            j = 1;
            while i <= n
                psth(i : i + window_size - 1) = ...
                    sum(spks(j : j + batch_size - 1)) / batch_size_sec;
                
                i = i + window_size;
                j = j + batch_size;
            end
            
            % smooth psth
            % - length of kernel
            kernel_size = floor(kernel_size_sec / dt_sec);
            % - 3 * sigma is enough (kernel_size = 6 * sigma)
            sigma = kernel_size / 6;
            
            psth = conv(...
                psth, ...
                fspecial('gaussian', [kernel_size, 1], sigma), ...
                'same' ...
            );
        end
        function mkdata(opts)
            %MKDATA make db from continuous 'spk' and 'vstim'.
            %[block diagram](./mkdata.vsdx)
            %
            %   Parameters
            %   ----------
            %   - opts : struct
            %      path of schema is './mkdata_schema.json'
            
            % inputs
            % - spk
            spk = getfield(load(opts.inputs.path), 'spk');
            % - vstim
            vstim = getfield(load(opts.inputs.path), 'vstim');
            
            % divide
            % - spks
            spks = DataUtils.divide_timeseries(...
                spk, ...
                opts.params.dt, ...
                opts.params.divide.trial_length ...
                );
            % - vstims
            vstims = DataUtils.divide_timeseries(...
                vstim, ...
                opts.params.dt, ...
                opts.params.divide.trial_length ...
                );
            
            % psth
            psth = DataUtils.spks_to_psth(...
                spks, ...
                opts.params.psth.window_size, ...
                opts.params.psth.kernel_size, ...
                opts.params.dt ...
                );
            
            figure();
            plot(psth);
            grid('on');
            title(...
                sprintf(...
                    'PSTH (window: %.3f, kernel: %.3f)', ...
                    opts.params.psth.window_size, ...
                    opts.params.psth.kernel_size ...
                ) ...
            );
            % vstim
            vstim = vstims(1, :);
            
            figure();
            plot(vstim);
            grid('on');
            title('VSTIM');
            
            % db
            % - length of each output sample
            sample_size = floor(opts.params.db.output_size / opts.params.dt);
            % - delta between two output samples
            delta_size = floor(...
                (opts.params.db.output_size - opts.params.db.output_intersection) / ...
                opts.params.dt ...
            );
            
            % - divide psth
            psths = DataUtils.divide_to_subvectors(...
                psth, ...
                sample_size, ...
                delta_size ...
            );
            
            % - divide vstim
            vstims = DataUtils.divide_to_subvectors(...
                vstim, ...
                sample_size, ...
                delta_size ...
            );
            % - make db
            db.x = num2cell(vstims', 1)';
            db.y = num2cell(psths', 1)';
            % - save db
            save(opts.outputs.path, 'db');
        end
        function mkdata_test()
            opts_path = './mkdata.json';
            DataUtils.mkdata(jsondecode(fileread(opts_path)));
        end
    end
end
