# NeuralNet project
This is my personal attempt at building a neural network, in ruby, you can experiment
with it if you want. 

You can make a new network just like i'm doing in the example in the lib/NeuralNet.rb
file with Network.new. The first argument is the width of the input layer, the second one is for the number of
hidden layers, after that comes an options array where you can specify a lot more.
If you want to make the network actually do something you will also need to specify at least the width of
the hidden layers, the matrix with the inputs and the matrix with the corresponding outputs.

The NMatrix library is used for all matrix calculations.

To train it simply use the train method on your network, with as the only argument the amount of training
cycles.
After you've trained the network you can use it on your data with the calculate method and an input matrix containing your data. And it will return a matrix with its solutions.
