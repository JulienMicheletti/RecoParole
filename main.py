#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.fft as FFT
import sys
import matplotlib.pyplot as plt
import math as math
from scipy.io.wavfile import read
from scipy.io.wavfile import write

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

def spectreamplitude(spectre, fftsize):
    return np.abs(spectre)

def spectrereconstruction(spectre_amplitude, spectre_phase, fftsize):
    test = []
    for i in range(0, fftsize):
        test.append(spectre_amplitude[i] * np.cos(spectre_phase[i]) + 1j * spectre_amplitude[i] * np.sin(spectre_phase[i]))
    return test

def soustractionspectrale(spectre_amplitude, moyenne):
    alpha = 2
    beta = 10
    gamma = 0
    sous = []
    for i in range(0, len(spectre_amplitude)):
        diff = np.power(spectre_amplitude[i], alpha) - beta * np.power(moyenne[i], alpha)
        if diff > 0:
            sous.append(np.power(np.power(spectre_amplitude[i], alpha) - beta * np.power(moyenne[i], alpha), 1/float(alpha)))

        else:
            sous.append(gamma * moyenne[i])
    return sous

def boucle_ola(signal, m, N):
    hamming = fenetrageHamming(N)
    tab_signal = np.empty(len(signal), dtype=float)
    somme_hamming = np.empty(len(signal), dtype=float)
    signal_fen = []
    spectre = []
    tabspectre_log = []
    tabspectre = []
    tabphase = []
    spectre = []
    spectre_debruite = []
    s_amplitude = []
    reconstruction = []
    bruit = np.empty(1024, dtype=complex)
    moyenne = np.empty(1024, dtype=complex)
    compt_moyenne = 0
    for i in range(0, len(signal) - N, m):
        signal_fen = signal[i:i+N] * hamming
        #INSERT HERE
        spectre = FFT.fft(signal_fen, 1024)
        s_amplitude = spectreamplitude(spectre, 1024)
        tabphase = spectrephase(spectre, 1024)
        tabspectre_log = 20 * np.log(s_amplitude)

        if compt_moyenne < 5 :
            bruit += s_amplitude
            moyenne = bruit / (compt_moyenne + 1)

        spectre_debruite = soustractionspectrale(s_amplitude, moyenne)
        reconstruction = spectrereconstruction(spectre_debruite, tabphase, 1024)

        spectre = FFT.ifft(reconstruction, 1024)
        signal_fen = np.real(spectre[0:N])
        #FIN INSERT
        tab_signal[i:i+N] += signal_fen
        somme_hamming[i:i+N] += hamming
        compt_moyenne += 1
    for i in range(0, len(tab_signal)):
        if somme_hamming[i] > 1e-08:
            tab_signal[i] /= somme_hamming[i]
    return tab_signal

if __name__ == "__main__":
    filename = sys.argv[1]
    data = read(filename)
    moyenne = np.mean(data[1])
    if moyenne != 0:
        for i in range(0, len(data[1])):
            data[1][i] -= moyenne
    fs = data[0]
    print("frequence echantillonnage", data[0])
    print("taille en echantillons", data[1].size)
    print("taille en ms", (data[1].size/data[0]) * 1000)
    m = 8 * data[0] / 1000
    N = 32 * data[0] / 1000

    plt.figure(1, figsize=(10,10))
    sigfig = plt.subplot(2,1,1)
    plt.plot(data[1])
    plt.xticks(np.arange(0, data[1].size, step=data[1].size/10))
    plt.xlabel('Temps (echt)')

    sig = boucle_ola(data[1], m, N)

    plt.subplot(2,1,2)
    plt.plot(sig)
    plt.show()

    write("resultat.wav", fs, np.int16(sig))
