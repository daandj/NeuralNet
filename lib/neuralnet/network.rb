class Network
  attr_accessor :number_inputs, :training_input, :training_output, :syn0,
  :calculation_input, :number_hiddenlayers, :width_hiddenlayer

  def initialize(number_inputs, number_hiddenlayers, opts = {})
    @number_inputs = number_inputs
    @number_hiddenlayers = number_hiddenlayers
    @training_input = opts[:training_input]
    @training_output = opts[:training_output]
    @calculation_input = opts[:calculation_input]
    @width_hiddenlayer = opts[:width_hiddenlayer] || @number_inputs
    # create synapse array
    init_syn
    # create layer arrays
    @l = []
    @l_error = []
    @l_delta = []
  end

  def train(iterations)
    (0..iterations).each do
      forward_prop(@training_input)
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
        @number_hiddenlayers.downto(1) do |curr_l|
          # how much did we miss?
          @l_error[curr_l] = @l_delta[curr_l+1].dot(@syn[curr_l].transpose)
          # multiply how much we missed by the
          # slope of the sigmoid at the values in l1
          @l_delta[curr_l] = @l_error[curr_l] * nonlin(@l[curr_l], true)
          # puts "l1, l2_delta: " + @l[curr_l].shape.to_s + @l_delta[curr_l+1].shape.to_s
          # update weights
          @syn[curr_l-1] += @l[curr_l-1].transpose.dot @l_delta[curr_l]
        end
      end
    end
  end

  def result
    puts @syn[0]
    puts @syn[1]
  end

  def calculate(input = @calculation_input)
    forward_prop(input)
    return @l.last
  end

  private

  def nonlin(x, deriv=false)
    if deriv
      return x*(-x + 1)
    else
      return ((-x).exp + 1) ** -1
    end
  end

  # initializes all the weights in the synapses randomly
  def init_syn
    @syn = []
    # First initialize the first synapse connecting the inputlayer and the first
    # hidden layer.
    @syn[0] = N.random([@number_inputs, @width_hiddenlayer]) * 2 - 1
    # Secondly initialize all the synapses connecting the hiddenlayers together.
    (1...@number_hiddenlayers).each do |number|
      @syn[number] = N.random([@width_hiddenlayer, @width_hiddenlayer]) * 2 - 1
    end
    # Finnaly initialize the synapse connecting the last hiddenlayer with the
    # output layer.
    @syn[@number_hiddenlayers] = N.random([@width_hiddenlayer, 1]) * 2 - 1
  end

  # Does an entire pass through the network
  def forward_prop(input)
    @l[0] = input

    (0..@number_hiddenlayers).each do |curr_l|
      # forward propagation
      @l[curr_l+1] = nonlin(@l[curr_l].dot @syn[curr_l])
    end
  end

end
