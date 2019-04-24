#!/usr/bin/python
import numpy as np
import soundfile as sf
import random as rnd
import math as m
import multiprocessing as mp

# Wの初期化

def score(x):
    return m.tanh(x)

def calc_partial_sum(cfg):
    wav, begin, end = cfg
    entropy = np.zeros([2, 2])
    for t in range(begin, end):
        phi = np.array([
            [score(wav[0][t]) * wav[0][t], score(wav[0][t]) * wav[1][t]],
            [score(wav[1][t]) * wav[0][t], score(wav[1][t]) * wav[1][t]]])
        entropy = entropy + phi
    return entropy

def entropy(wav):
    cpu_num = mp.cpu_count()
    step = int(m.ceil(len(ipt[0]) / cpu_num))
    pool = mp.Pool(cpu_num)

    cfgs = []
    for i in range(0, cpu_num):
        cfgs.append((wav, i * step, min((i+1) * step, len(ipt[0]) - 1)))

    entropy = np.zeros([2, 2])
    for sub in pool.map(calc_partial_sum, cfgs):
        entropy = entropy + sub
    pool.close()
    return entropy / len(ipt[0])


if __name__ == '__main__':
    wav1, rate1 = sf.read('mix1.wav')
    wav2, rate2 = sf.read('mix2.wav')
    ipt = np.array([wav1, wav2])
    out = np.array([wav1, wav2])
    step = 0.2
    w = np.array([[rnd.random() - 0.5, rnd.random() - 0.5], [rnd.random() - 0.5, rnd.random() - 0.5]])

    for i in range(0, 100):
        e = entropy(out)
        print(e)
        w = w + step * (np.identity(2) - e) @ w
        out = w @ ipt

    sf.write('ica1.wav', out[0], rate1)
    sf.write('ica2.wav', out[1], rate2)
