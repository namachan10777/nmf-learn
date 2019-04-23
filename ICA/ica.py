#!/usr/bin/python
import numpy as np
import soundfile as sf
import random as rnd
import math as m

# Wの初期化

def laplace(phi, mu, x):
    return m.exp(-abs(x-mu)/phi) / phi / 2

def score(x):
    return laplace(0.5, 0.0, x)

if __name__ == '__main__':
    wav1, rate1 = sf.read('mix1.wav')
    wav2, rate2 = sf.read('mix2.wav')
    w = np.array([[rnd.random() - 0.5, rnd.random() - 0.5], [rnd.random() - 0.5, rnd.random() - 0.5]])
    step = 0.1
    for i in range(0, 10):
        entropy = np.array([[0., 0.], [0., 0.]])
        for t in range(0, len(wav1)):
            phi = np.array([
                [score(wav1[t]) * wav1[t], score(wav1[t]) * wav2[t]],
                [score(wav2[t]) * wav1[t], score(wav2[t]) * wav2[t]]])
            entropy = np.add(entropy, phi)
        np.divide(entropy, t)
    w = w + np.dot((np.array([[1, 0], [0, 1]]) - entropy) * step, w)

    input_matrix = np.array([wav1, wav2])
    out = np.dot(w, input_matrix)
    sf.write('ica1.wav', out[0], rate1)
    sf.write('ica2.wav', out[1], rate2)
