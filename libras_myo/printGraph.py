#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from segmentData import readCsv, calculateAverageEnergy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import column_or_1d
from mpl_toolkits.mplot3d import Axes3D


def readCsvSegmented(inputFileName):

    data = np.genfromtxt(inputFileName, delimiter=',')
    return data


def print3D(X,labels):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap=plt.cm.get_cmap('rainbow'))
    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')
    ax.set_zlabel('LD 3')

    plt.show()


def printGraph(X,y):
    # data = np.genfromtxt(filepath, delimiter=',', dtype=None)
    # data = sequence[:, 1:]
    # y = y[None].T
    y = np.array(y)[None].T


    # classes = classes[...,None]
    # print sequence[:,0]
    # lda = LDA(n_components=2)
    # sequence = lda.fit_transform(X, y)
    # print sequence
    # sequence = np.append(sequence[:,0],classes, 1)
    # classes = y[:,0][None].T
    # print data
    # y = column_or_1d(y, warn=True)
    plt.axis([-100,100,-100,100])
    data = np.append(X, y, 1)
    print data
    plt.grid(True)
    for x,y,c in data:
        color = 'b'
        if c == 0:
            color = 'r'
        elif c == 1:
            color = 'b'
        elif c == 2:
            color = 'g'
        elif c== 3:
            color = 'm'
        elif c==4:
            color = 'c'
        else:
            color = 'k'

        plt.scatter(x,y, color=color)


    # plt.plot(sequence)
    # plt.savefig("teste.png")
    plt.show()


if __name__ == '__main__':
    f1 = readCsv('data/bernardo/bernardo-C-1-emg.csv')

    # f2 = readCsvSegmented('data_segmented/sylvia/sylvia-A-2-emg-segmented.csv')
    f3 = calculateAverageEnergy(f1)
    printGraph(f3)
    # printGraph(f2)
    # printGraph(f3)
