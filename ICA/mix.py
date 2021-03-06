#!/usr/bin/python
# 2019 Nakano Masaki

import sys
import soundfile as sf
import numpy as np

if __name__ == '__main__':
    n = len(sys.args)
    mix = np.random.rand(n, n) * 2 - 1
    _, samplerate = sf.read(sys.args[1])
    wavs = []
    for f in sys.args:
        wav, rate = sf.read(f)
        assert rate, 'contradictory samplerate'
        wavs.append(wav)

    mixed = mix * np.array(wavs)


data1, samplrate1 =  sf.read('s1.wav')
data2, samplrate2 =  sf.read('s2.wav')

mix_matrix = np.array([[0.4, 0.6],[-0.8, 0.7]])
sounds = np.array([data1, data2])
print(sounds)
mix_sounds = np.dot(mix_matrix, sounds)

sf.write('mix1.wav', mix_sounds[0], samplrate1)
sf.write('mix2.wav', mix_sounds[1], samplrate2)
