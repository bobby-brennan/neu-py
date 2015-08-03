import numpy as Numpy
import theano as Theano
import math

min_char = 32
max_char = 122
special_chars = 2
num_chars = max_char - min_char + 1 + special_chars

def charToVec(c):
  code = ord(c)
  if c == '\n':
    code = 1
  elif (code < min_char or code > max_char):
    code = 0
  else:
    code = code - min_char + special_chars
  vec = Numpy.zeros((num_chars), dtype=Theano.config.floatX)
  vec[code] = 1.0
  return vec

def strToVec(str, skip=-1):
  sum = Numpy.zeros((num_chars), dtype=Theano.config.floatX)
  for i in range(0, len(str)):
    if skip == i:
      continue
    charVec = charToVec(str[i]) 
    weight = 1.0 / (1 + abs(len(str) / 2 - i))
    sum = Numpy.add(sum, weight * charVec)
  return sum

def vecToChar(vec):
  code = Numpy.argmax(vec)
  if (code == 0):
    return '~'
  elif (code == 1):
    return '\n'
  else:
    return chr(code + min_char - 2)

def vecsToStr(vecs):
  ret = ''
  for i in range(0, len(vecs)):
    ret = ret + vecToChar(vecs[i])
  return ret

