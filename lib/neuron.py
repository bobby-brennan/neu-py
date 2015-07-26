import numpy as np

class neuron:
  def __init__(self, options):
    self.weights = {};
    self.weights['hh'] = np.random.rand(options.h_size, options.h_size)
    self.weights['xh'] = np.random.rand(options.x_size, options.h_size)
    self.weights['hy'] = np.random.rand(options.h_size, options.y_size)
    self.h = np.zeros(options.h_size);

  def probe(self, input, isTest=false):
    history = np.dot(self.weights['hh'], self.h)
    info = np.dot(input, self.weights['xh'])
    sum = np.add(history, info)
    nextH = np.tanh(sum)
    output = np.dot(nextH, self.weights['hy'])
    if !isTest:
      self.nextH = nextH
    return output

  def train(self, input, output, expected):
    loss = self.getLoss(output, expected)
    for name, weights in self.weights.items():
      randAdjust = np.random.rand(weights.shape[0], weights.shape[1])
      self.weights[name] = map(adjust, weights) 

  def step(self):
    self.h = self.nextH
  
