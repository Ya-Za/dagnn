classdef CrossEntropy < dagnn.Loss
  properties
      epsilon = 1e-6;
  end

  methods
    function outputs = forward(obj, inputs, params)
        % y^ := inputs{1}
        % y  := inputs{2}
        % N = size(inputs{1}, 1) * size(inputs{1}, 2) * size(inputs{1}, 3);
        yhat = inputs{1}(:);
        y = inputs{2}(:);
        
        outputs{1} = -1 * (...
            dot(y, log(max(yhat, obj.epsilon))) + ...
            dot((1 - y), log(max(1 - yhat, obj.epsilon))) ...
        );
    
        obj.accumulateAverage(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % y^ := inputs{1}
        % y  := inputs{2}
        % N = size(inputs{1}, 1) * size(inputs{1}, 2) * size(inputs{1}, 3);
        
%         derInputs{1} = derOutputs{1} .* (...
%             ((1 - inputs{2}) ./ max((1 - inputs{1}), obj.epsilon)) - ...
%             (inputs{2} ./ max(inputs{1}, obj.epsilon)) ...
%         );
    
        derInputs{1} = derOutputs{1} .* (...
            (inputs{1} - inputs{2}) ./ max(inputs{1} .* (1 - inputs{1}), obj.epsilon) ...
        );
    
        derInputs{2} = derOutputs{1} .* (...
            log(max((1 - inputs{1}) ./ max(inputs{1}, obj.epsilon), obj.epsilon)) ...
        );
    
        derParams = {};
    end

    function obj = CrossEntropy(varargin)
      obj.load(varargin);
      obj.loss = 'crossentropy';
    end
  end
end
