class Network
  attr_accessor :number_inputs, :training_input, :training_output, :syn0,
  :calculation_input, :number_hiddenlayers

  def initialize(number_inputs, number_hiddenlayers, opts = {})
    @number_inputs = number_inputs
    @number_hiddenlayers = number_hiddenlayers
    @training_input = opts[:training_input]
    @training_output = opts[:training_output]
    @calculation_input = opts[:calculation_input]
    # create synapse array
    @syn = []
    # create layer arrays
    @l = []
    @l_error = []
    @l_delta = []
    # initialize weights randomly
    (0..@number_hiddenlayers).each do |number|
      @syn[number] = N.random([@number_inputs, 1]) * 2 - 1
    end
  end

  def train(iterations)
    (0..iterations).each do |x|
      @l[0] = @training_input

      # an entire pass through the network
      (0..@number_hiddenlayers).each do |curr_l|
        # forward propagation
        @l[curr_l+1] = nonlin(@l[curr_l].dot @syn[curr_l])
      end

      # how much did we mis the target output?
      @l_error[@number_hiddenlayers+1] = @training_output - @l[@number_hiddenlayers+1]

      # multiply how much we missed by the
      # slope of the sigmoid at the values in l1
      @l_delta[@number_hiddenlayers+1] = @l_error[@number_hiddenlayers+1] *
                                      nonlin(@l[@number_hiddenlayers+1], true)

      # update the synapse connecting the last two layers
      @syn[@number_hiddenlayers] += @l[@number_hiddenlayers].transpose.dot @l_delta[@number_hiddenlayers+1]

      # back propagation for the remaining layers
      if @number_hiddenlayers > 0
        (@number_hiddenlayers..0).each do |curr_l|
          # how much did we miss?
          @l_error[curr_l] = @l_delta[curr_l+1].dot @syn[curr_l].transpose
          # multiply how much we missed by the
          # slope of the sigmoid at the values in l1
          @l_delta[curr_l] = @l_error[curr_l] * nonlin(@l[curr_l], true)
          # update weights
          @syn[curr_l] += @l[curr_l].transpose.dot @l_delta[curr_l]
        end
      end
    end
  end

  def result
    @syn[0]
  end

  def calculate(l0 = @calculation_input)
    l1 = nonlin(l0.dot @syn[0])
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
