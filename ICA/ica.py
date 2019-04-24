#!/usr/bin/python
import numpy as np
import soundfile as sf
import random as rnd
import math as m

# Wの初期化

def laplace(phi, mu, x):
    return m.exp(-abs(x-mu)/phi) / phi / 2

def score(x):
    #return laplace(1.0, 0.0, x)
    if x == 0:
        return 0
    return x / abs(x)

if __name__ == '__main__':
    wav1, rate1 = sf.read('mix1.wav')
    wav2, rate2 = sf.read('mix2.wav')
    ipt = np.array([wav1, wav2])
    out = np.array([wav1, wav2])
    step = 0.2
    w = np.array([[rnd.random() - 0.5, rnd.random() - 0.5], [rnd.random() - 0.5, rnd.random() - 0.5]])

    for i in range(0, 100):
        entropy = np.zeros([2, 2])
        for t in range(0, len(wav1)):
            phi = np.array([
                [score(out[0][t]) * out[0][t], score(out[0][t]) * out[1][t]],
                [score(out[1][t]) * out[0][t], score(out[1][t]) * out[1][t]]])
            entropy = np.add(entropy, phi)
        entropy = entropy / len(wav1)
        print(entropy)
        w = w + step * (np.identity(2) - entropy) @ w
        out = w @ ipt

    sf.write('ica1.wav', out[0], rate1)
    sf.write('ica2.wav', out[1], rate2)
