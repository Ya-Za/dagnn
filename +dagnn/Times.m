classdef Times < dagnn.ElementWise
  properties
  end

  methods
    function outputs = forward(obj, inputs, params)
        outputs{1} = inputs{1} .* inputs{2};
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs{1} = derOutputs{1} .* inputs{2};
        derInputs{2} = derOutputs{1} .* inputs{1};
        
        derParams = {} ;
    end

    function obj = Times(varargin)
      obj.load(varargin) ;
    end
  end
end
