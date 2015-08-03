import lib.neuron as Neuron
import util.text_util as Text
import numpy as Numpy
import theano as Theano
import math
import sys

x_size = Text.num_chars
y_size = Text.num_chars

window_size = 15
neurons = []
for i in range(0, window_size):
  neurons.append(Neuron.Neuron(x_size, y_size, h_size=window_size, rate=.05, range=.05))

text = open('./data/shak-mini.txt')

charBuf = text.read(window_size)
errs = [0] * 100
avgErr = 0.0
total = 0
while True:
  output = ''
  for i in range(0, window_size):
    input = Text.strToVec(charBuf, i)
    expected = Text.charToVec(charBuf[i])
    h, o, e = neurons[i].train(input, expected)
    output = output + Text.vecToChar(o)
  
  total = total + 1 
  if (total % 500 < 5):
    print 'IN: ' + charBuf.replace('\n', '$')
    print 'OUT:' + output.replace('\n', '$')
    print e
    if (total % 500 == 4): print ''

  next = text.read(1)
  if not next:
    break
  charBuf = charBuf[1:] + next

print
print
print '------------'
print
print

char_predictions = []
for i in range(0, window_size):
  char_predictions.append(Text.charToVec(charBuf[i]))

gen_text = ''
for i in range(0, 1000):
  for n in range(window_size - 1, -1, -1):
    prediction = Text.vecsToStr(char_predictions)
    out = neurons[n].step(Text.strToVec(prediction, n))
    char_predictions[n] = char_predictions[n] + out
  prediction = Text.vecsToStr(char_predictions)
  gen_text = gen_text + prediction[0]
  char_predictions = char_predictions[1:]
  char_predictions.append(Numpy.zeros((x_size), dtype=Theano.config.floatX))

print
print
print '------------'
print
print

print gen_text
