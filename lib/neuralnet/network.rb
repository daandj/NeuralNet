class Network
  attr_accessor :number_inputs, :training_input, :training_output, :syn0, :calculation_input

  def initialize(number_inputs, opts = {})
    @number_inputs = number_inputs
    @training_input = opts[:training_input]
    @training_output = opts[:training_output]
    @calculation_input = opts[:calculation_input]

    # initialize weights randomly
    @syn0 = N.random([@number_inputs, 1]) * 2 - 1
  end

  def train(iterations)
    (0..iterations).each do
      # forward propagation
      l0 = @training_input
      l1 = nonlin(l0.dot @syn0)

      # how much did we miss?
      l1_error = @training_output - l1

      # multiply how much we missed by the
      # slope of the sigmoid at the values in l1
      l1_delta = l1_error * nonlin(l1, true)

      # update weights
      @syn0 += l0.transpose.dot l1_delta
    end
  end

  def result
    @syn0
  end

  def calculate(l0 = @calculation_input)
    l1 = nonlin(l0.dot @syn0)
    return l1
  end

  private

  def nonlin(x, deriv=false)
    if deriv
      return x*(-x + 1)
    else
      return ((-x).exp + 1) ** -1
    end
  end

end
