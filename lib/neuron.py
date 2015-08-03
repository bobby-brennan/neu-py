import theano as Theano
import theano.tensor as T
import numpy as Numpy

class Neuron:
  def __init__(self, x_size, y_size, h_size, rate, range=1):
    x = T.dvector('x')

    self.h = Theano.shared(
        value=Numpy.zeros((h_size)),
        name='h')

    w_hh = Theano.shared(
        value=Numpy.random.uniform(size=(h_size, h_size), low=-range, high=range),
        name='w_hh')
    w_xh = Theano.shared(
        value=Numpy.random.uniform(size=(x_size, h_size), low=-range, high=range),
        name='w_xh')
    w_hy = Theano.shared(
        value=Numpy.random.uniform(size=(h_size, y_size), low=-range, high=range),
        name='w_hy')

    history = T.dot(self.h, w_hh)
    input = T.dot(x, w_xh)
    sum = T.add(history, input)
    new_h = T.tanh(sum)
    output = T.dot(new_h, w_hy)
    #output = output / T.sum(output)

    target = T.dvector('target')
    error = T.sum((output - target) ** 2)
    #error = (T.dot(target, output) / (T.sqrt(T.sum(target) ** 2) * T.sqrt(T.sum(output) ** 2)) * -1 + 1) / 2
    #error = T.dot(output, target)
    g_w_hh, g_w_xh, g_w_hy = T.grad(error, [w_hh, w_xh, w_hy])

    self.step = Theano.function([x], output,
        updates=[(self.h, new_h)])

    self.train = Theano.function([x, target], [new_h, output, error],
      updates=[(self.h, new_h),
               (w_hh, w_hh - rate * g_w_hh),
               (w_xh, w_xh - rate * g_w_xh),
               (w_hy, w_hy - rate * g_w_hy)])

  def addSample(self, sample, target):
    return self.train(sample, target)

  def takeStep(self, sample):
    return self.step(sample)

  def resetHistory(self):
    size = self.h.get_value().size
    self.h.set_value(Numpy.zeros(size))
