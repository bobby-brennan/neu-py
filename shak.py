import lib.neuron as Neuron
import numpy as Numpy
import theano as Theano

neuron = Neuron.Neuron(x_size=256, y_size=256, h_size=100, rate=.001)

text = open('./data/shak.txt')

def charToVec(c):
  code = ord(c)
  vec = Numpy.zeros((256), dtype=Theano.config.floatX)
  vec[code] = 1.0
  return vec

c = charToVec(text.read(1))
nextC = charToVec(text.read(1))
print 'c:{}'.format(c);
print 'n:{}'.format(nextC);
avgErr = 0.0
total = 0
while True:
  h, o, e = neuron.train(c, nextC)
  avgErr = .99 * avgErr + .01 * e;
  total = total + 1
  if (total % 1000 == 1):
    print 'err:{}'.format(avgErr)
  c = nextC
  nextC = text.read(1)
  if not nextC:
    break
  nextC = charToVec(nextC)
