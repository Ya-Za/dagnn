classdef TanSigmoid < dagnn.ElementWise
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = 2 * vl_nnsigmoid(2 * inputs{1}) - 1;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = 4 * vl_nnsigmoid(2 * inputs{1}, derOutputs{1}) ;
      derParams = {} ;
    end
  end
end
