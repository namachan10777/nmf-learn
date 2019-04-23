#!/usr/bin/python
import numpy as np
import soundfiles as sf
import random as rnd

# Wの初期化
w = np.array([[rnd.random() - 0.5, rnd.random() - 0.5], [rnd.random() - 0.5, rnd.random() - 0.5]])

def laplace(phi, mu, x):
    return exp(-abs(x-mu)/phi) / phi / 2

def score(x):
    return laplace(0.5, 0.0, x)

if __name__ == '__main__':
    wav1, rate1 = sf.read('mix1.wav')
    wav2, rate2 = sf.read('mix2.wav')
    for i as range(0, 1000):
        entropy = np.array([0., 0.], [0., 0.])
        for t as range(0, len(wav1)):
            phi = np.array([
                [score(wav1[t]) * wav1[t], score(wav1[t]) * wav2[t]],
                [score(wav2[t]) * wav1[t], score(wav2[t]) * wav2[t]]])
            entropy = np.add(entropy, phi)
        np.div(entropy, t)

