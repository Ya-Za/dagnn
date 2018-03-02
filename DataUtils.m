classdef DataUtils < handle
    %DATAUTILS contains data-utilities functions
    
    methods (Static)
        function output = divide_to_subvectors( input, l, d )
            %DIVIDE_TO_SUBVECTORS splits input vector to subvectors
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
        function output = divide_timeseries( input, dt_sec, l_sec, d_sec)
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
        function output = divide_to_samelength_subvectors( input, m )
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
        function output = downsample_vector( input, m )
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
        function output = resize( input, output_size, method)
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
