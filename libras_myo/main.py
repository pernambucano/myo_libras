#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from segmentData import calculateAverageEnergy, segmentAveragedSignal, readCsv, writeDataInFile, segmentSensorsData
import numpy as np
import csv
from printGraph import printGraph
from calculateFeatures import getFeatures
import matplotlib.pyplot as plt
from classification import classify


user = "paulo"
INPUT_TEST_WITHOUT_USER = "data/%s/%s-Teste_sem_usuario" % (user, user)  # 3 x
INPUT_TEST_TR = "data/%s/%s-Teste_mao_esticada" % (user, user)  # 3 x
INPUT_A = "data/%s/%s-A_perto_cotovelo" % (user, user)  # x 3
INPUT_B = "data/%s/%s-B_perto_cotovelo" % (user, user)  # x 3
INPUT_C = "data/%s/%s-C_perto_cotovelo" % (user, user)  # x 3
INPUT_D = "data/%s/%s-D_perto_cotovelo" % (user, user)  # x 3
INPUT_E = "data/%s/%s-E_perto_cotovelo" % (user, user)  # x 3
OUTPUT = "data_segmented/%s" % user


INPUT_A_SEGMENTED = "data_segmented/%s/%s-A_perto_cotovelo" % (user, user)
INPUT_B_SEGMENTED = "data_segmented/%s/%s-B_perto_cotovelo" % (user, user)
INPUT_C_SEGMENTED = "data_segmented/%s/%s-C_perto_cotovelo" % (user, user)
INPUT_D_SEGMENTED = "data_segmented/%s/%s-D_perto_cotovelo" % (user, user)
INPUT_E_SEGMENTED = "data_segmented/%s/%s-E_perto_cotovelo" % (user, user)


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
    segmentFiles()
    featureMatrix = getAttributes()
    print classify(featureMatrix)
