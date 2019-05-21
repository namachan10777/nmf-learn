#!/usr/bin/python
import numpy as np
import math as m
import soundfile as sf
import matplotlib.pyplot as plot

def genwin(winsize, f):
    win = np.zeros(winsize)
    for i in range(0, winsize):
        y = f(i/winsize)
        win[i] = y
    return win

def hann(winsize):
    return genwin(winsize, lambda x: 0.5 - 0.5 * m.cos(2 * m.pi * x))

def hamming(winsize):
    return genwin(winsize, lambda x: 0.54 - 0.46 * m.cos(2 * m.pi * x))

def dft(wav, winsize=200, stride_divide=2, wintype='hamming'):
    stride = int(winsize/stride_divide)
    iter_times = int(m.ceil(len(wav) / winsize)) * stride_divide
    target = np.zeros((m.ceil(len(wav) / winsize) + 1) * winsize)
    target[int(winsize/2):len(wav)+int(winsize/2)] = np.array(wav)
    result = np.zeros([iter_times, winsize])
    if wintype == 'hann':
        win = hann(winsize)
    elif wintype == 'hamming':
        win = hamming(winsize)

    for i in range(0, iter_times):
        source = target[i*stride:i*stride+winsize] * win
        result[i] = 20 * np.sqrt(np.abs(np.fft.fft(source)))
    plot.imshow(result.T)
    plot.show()


if __name__ == '__main__':
    wav, rate = sf.read('s1.wav')
    dft(wav, winsize=200, stride_divide=8)
