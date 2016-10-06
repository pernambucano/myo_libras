import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sequence = pd.read_csv('dados-emg.csv', sep=',', header=None)
sequence = sequence.iloc[:, 1:]
# plt.xticks([x for x in xrange(0,27)])
plt.grid(True)
# plt.title('Letra A Segmentada')
# plt.xlabel('windows')
# plt.ylabel('energy')
plt.plot(sequence)
plt.savefig("teste.png")
