import theano as Theano
import theano.tensor as T
import numpy as Numpy

class Net:
  def __init__(self, x_size, y_size, h_size, rate, length):
    self.neurons = []
    for i in range(1, length):
      x = i == 1 ? T.dvector('x') : self.neurons[i - 2].output
      neuron = Neuron.Neuron(x_size, y_size, h_size, rate, x)
      self.neurons.append(neuron)

  def addSample(input, target):
    
    for neuron in self.neurons:
      
