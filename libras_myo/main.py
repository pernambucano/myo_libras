#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from segmentData import calculateAverageEnergy, segmentAveragedSignal, readCsv, writeDataInFile, segmentSensorsData, segmentContinuousData
import numpy as np
import csv
from printGraph import printGraph
from calculateFeatures import getFeatures
import matplotlib.pyplot as plt
from classification import classify



letters = ['A', 'B', 'C', 'D', 'E']
users = ['karla', 'paulo']
directory = "data"
directorySegmented = "data_segmented"
desc = "perto_cotovelo"


def readCsvSegmented(inputFileName):

    data = np.genfromtxt(inputFileName, delimiter=',', dtype=None)
    return data


def calculateTreshold(user):
    meanAveragedEmg = 0
    for turn in xrange(1, 4):
        # input_test_tr_complete = '%s-%d-emg.csv' % (INPUT_TEST_TR, turn)
        path = "%s/%s/%s-Teste_mao_esticada-%s-emg.csv" % (directory, user, user, turn)
        print path
        data = readCsv(path)
        averagedEmg = calculateAverageEnergy(data)
        meanAveragedEmg += np.amax(averagedEmg)

    meanAveragedEmg = meanAveragedEmg / 3
    threshold = meanAveragedEmg * 0.05
    return threshold


def segmentFile(path, user):
    threshold = calculateTreshold(user)
    data = readCsv(path)
    averagedEmg = calculateAverageEnergy(data)
    letterIndexes = segmentContinuousData(averagedEmg)

    # TODO: Segmentar as letras / melhorar
    print letterIndexes
    print len(letterIndexes)


def segmentFiles():


    for user in users:
        threshold = calculateTreshold(user) # Calcula o  threshold de cada usuário
        for turn in xrange(1, 4):
            for letter in letters:
                path = "%s/%s/%s-%s_%s-%s-emg.csv" % (directory, user, user, letter, desc, turn)

                print path

                # Get data
                data = readCsv(path)

                # Segment Data by Average EMG
                averageEmg = calculateAverageEnergy(data)

                startOfLetter, endOfLetter = segmentAveragedSignal(
                averageEmg, threshold)

                data_segmented = segmentSensorsData(
                data, startOfLetter, endOfLetter)

                # # Save data in specific folder
                path_segmented = "%s/%s/%s-%s_%s-%s-emg-segmented.csv" % (directorySegmented, user, user, letter, desc, turn)

                writeDataInFile(data, path_segmented)


def getAttributes():

    dataset = []
    for user in users:
        for turn in xrange(1, 4):
            classLetter = 0
            for letter in letters:

                path_segmented = "%s/%s/%s-%s_%s-%s-emg-segmented.csv" % (directorySegmented, user, user, letter, desc, turn)

                # Get data
                data = readCsvSegmented(path_segmented)

                # Get features
                dataset.append(getFeatures(data, classLetter).tolist())
                print user, turn, letter, classLetter
                classLetter += 1

    ndataset = np.array(dataset)
    # print ndataset[:,64]
    return ndataset


if __name__ == '__main__':
    # segmentFiles()
    # featureMatrix = getAttributes()
    # print classify(featureMatrix)
    segmentFile("data/karla/karla-Alfabeto_inteiro-1-emg.csv", "karla")
