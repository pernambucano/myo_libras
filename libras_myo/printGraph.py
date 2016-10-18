#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# sequence = pd.read_csv(, sep=',', header=None)
data = np.genfromtxt('data/karla/karla-Alfabeto_inteiro-1-emg.csv', delimiter=',', dtype=None)
# sequence = sequence.iloc[:, 1:]
print data
data = data[:, 1:]
# plt.xticks([x for x in xrange(0,27)])
plt.grid(True)
# plt.title('Letra A Segmentada')
# plt.xlabel('windows')
# plt.ylabel('energy')
plt.plot(data)
plt.savefig("teste.png")
