require 'NMatrix'
require 'benchmark'

# Sigmoid function
def nonlin(x, deriv=false)
  if deriv
    return x*(-x + 1)
  else
    return ((-x).exp + 1) ** -1
  end
end

# input dataset
x = N[  [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1], dtype: :float64 ]

# output dataset
y = N[[0,0,1,1], dtype: :float64].transpose

# seed random numbers to make calculation
# deterministic (just ad good practice)
srand(1)

# initialize weights randomly with mean 0
syn0 = N.random([3, 1]) * 2 - 1
# puts "Synaps 0 before learning."
# puts syn0

(0..10000).each do
  # forward propagation
  l0 = x
  l1 = nonlin(l0.dot syn0)

  # how much did we miss?
  l1_error = y - l1

  # multiply how much we missed by the
  # slope of the sigmoid at the values in l1

  l1_delta = l1_error * nonlin(l1, true)

  # update weights
  syn0 += l0.transpose.dot l1_delta

end

puts "Output after training:"
puts nonlin(x.dot syn0)
