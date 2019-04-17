#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.fft as FFT
import math as math
from scipy.io.wavfile import read

def get_sign(value):
    if value > 0:
        return 1
    else:
        return -1

def fenetrageHamming(size):
    wn = []
    for i in range(0, size):
        wn.append(0.54 - 0.46 * math.cos(2 * math.pi * i / size))
    return wn

def spectrephase(spectre, fftsize):
    return np.angle(spectre)

def spectreamplitude(spectre, fftsize) :
    return np.abs(spectre)

def spectrereconstruction(spectre_amplitude, spectre_phase, fftsize):
    i = len(spectre_amplitude) - 1
    return spectre_amplitude[i] * np.cos(spectre_phase[i]) + 1j * spectre_amplitude[i] * np.sin(spectre_phase[i])

def boucle_ola(signal, m, N):
    hamming = fenetrageHamming(N)
    tab_signal = np.empty(len(signal), dtype=float)
    somme_hamming = np.empty(len(signal), dtype=float)
    signal_fen = []
    tabspectre = []
    tabphase = []
    spectre = []
    reconstruction = []
    moyenne = np.empty(1024, dtype=complex)
    for i in range(0, len(signal) - N, m):
        signal_fen = signal[i:i+N] * hamming
        #INSERT HERE
        spectre = FFT.fft(signal_fen, 1024)
        tabspectre.append(20 * spectreamplitude(spectre, 1024))
        tabphase.append(spectrephase(spectre, 1024))
        if i < 5 :
            moyenne += spectre
        reconstruction.append(spectrereconstruction(tabspectre, tabphase, 1024))
        signal_fen = FFT.ifft(spectre, 1024)
        signal_fen = np.real(spectre[0:N])
        #FIN INSERT
        tab_signal[i:i+N] += signal_fen
        somme_hamming[i:i+N] += hamming
    for i in range(0, len(tab_signal)):
        if somme_hamming[i] > 1e-08:
            tab_signal[i] /= somme_hamming[i]
    moyenne = moyenne / 5
    plt.plot(np.transpose(moyenne))
    return tab_signal


if __name__ == "__main__":
    data = read('bruit.wav')
    moyenne = np.mean(data[1])
    if moyenne != 0:
        for i in range(0, len(data[1])):
            data[1][i] -= moyenne
    print("frequence echantillonnage", data[0])
    print("taille en echantillons", data[1].size)
    print("taille en ms", (data[1].size/data[0]) * 1000)
    m = 8 * data[0] / 1000
    N = 32 * data[0] / 1000

    plt.figure(1, figsize=(10,10))
    sigfig = plt.subplot(3,1,1)
    plt.plot(data[1])
    plt.xticks(np.arange(0, data[1].size, step=data[1].size/10))
    plt.xlabel('Temps (echt)')

    plt.subplot(3,1,2)
    hamming = fenetrageHamming(N)
    plt.plot(hamming)
    ymin, ymax = plt.ylim()
    yborne = max(np.abs(ymin), np.abs(ymax))
    plt.yticks([0, yborne+0.5])

    sig = boucle_ola(data[1], m, N)
    plt.subplot(3,1,3)
    plt.plot(sig)
    plt.show()
