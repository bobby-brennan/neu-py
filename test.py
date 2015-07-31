import lib.neuron as Neuron

import unittest
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

TRIALS = 10000
MAX_ERR = .01

class PatternTest(unittest.TestCase):
  def test(self):
    for x in range(1, TRIALS):
      h, o, e = Neuron.addSample(SAMPLES[x % len(SAMPLES)], TARGETS[x % len(TARGETS)])
      #print('i:{}\tt:{}\to:{}\te:{}'.format(SAMPLES[x % len(SAMPLES)], TARGETS[x % len(TARGETS)], o, e))

    for x in range(TRIALS, TRIALS + len(SAMPLES) * 2):
      h, o, e = Neuron.addSample(SAMPLES[x % len(SAMPLES)], TARGETS[x % len(TARGETS)])
      print('i:{}\tt:{}\to:{}\te:{}'.format(SAMPLES[x % len(SAMPLES)], TARGETS[x % len(TARGETS)], o, e))
      self.assertEqual(True, e < MAX_ERR)


if __name__ == '__main__':
    unittest.main()
