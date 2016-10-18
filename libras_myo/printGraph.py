#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def printGraph(sequence):
    # data = np.genfromtxt(filepath, delimiter=',', dtype=None)
    # data = sequence[:, 1:]
    plt.grid(True)
    plt.plot(sequence)
    # plt.savefig("teste.png")
    plt.show()
