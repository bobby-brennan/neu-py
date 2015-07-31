import lib.neuron as Neuron

import numpy as Numpy

SAMPLES = [
  Numpy.array([-.3]),
  Numpy.array([.5]),
  Numpy.array([.8]),
  Numpy.array([.8]),
  Numpy.array([.8]),
]
TARGETS = [
  Numpy.array([.1]),
  Numpy.array([-.9])
]
for x in range(1, 10000):
  h, o, e = Neuron.addSample(SAMPLES[x % len(SAMPLES)], TARGETS[x % len(TARGETS)])
  print('i:{}\to:{}\te:{}'.format(SAMPLES[x % len(SAMPLES)], o, e))
