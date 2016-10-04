import matplotlib.pyplot as pyplot
import numpy as np

x = np.random.normal(0,0.1,1000)

pyplot.plot(x)
pyplot.savefig("teste.png")
