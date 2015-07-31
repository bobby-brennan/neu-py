import theano as Theano
import theano.tensor as T
import numpy as Numpy

x_size = 1
h_size = 6
y_size = 1
rate = .05

x = T.dvector('x')

h = Theano.shared(
    value=Numpy.zeros((h_size), dtype=Theano.config.floatX),
    name='h')

w_hh = Theano.shared(
    value=Numpy.random.uniform(size=(h_size, h_size), low=-1, high=1),
    name='w_hh')
w_xh = Theano.shared(
    value=Numpy.random.uniform(size=(x_size, h_size), low=-1, high=1),
    name='w_xh')
w_hy = Theano.shared(
    value=Numpy.random.uniform(size=(h_size, y_size), low=-1, high=1),
    name='w_hy')

history = T.dot(h, w_hh)
input = T.dot(x, w_xh)
sum = T.add(history, input)
new_h = T.tanh(sum)
output = T.dot(new_h, w_hy)

target = T.dvector('target')
error = T.sum((output - target) ** 2)
g_w_hh, g_w_xh, g_w_hy = T.grad(error, [w_hh, w_xh, w_hy])

step = Theano.function([x], output,
    updates=[(h, new_h)])

train = Theano.function([x, target], [new_h, output, error],
  updates=[(h, new_h),
           (w_hh, w_hh - rate * g_w_hh),
           (w_xh, w_xh - rate * g_w_xh),
           (w_hy, w_hy - rate * g_w_hy)])

def addSample(sample, target):
  return train(sample, target)

def takeStep(sample):
  return step(sample)
