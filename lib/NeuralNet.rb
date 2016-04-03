require_relative "neuralnet/network.rb"
require "NMatrix"

srand(1)

x = N[  [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1], dtype: :float64 ]

y = N[[0,0,1,1], dtype: :float64].transpose

test_set = N[ [0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1], dtype: :float64 ]

nn = Network.new(3, 0, training_input: x, training_output: y)
nn.train(10000)
puts nn.calculate(test_set)
puts nn.result
