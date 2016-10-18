#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from segmentData import calculateAverageEnergy, segmentSignal, readCsv, writeDataInFile
import numpy as np
import csv
from printGraph import printGraph


INPUT_TEST_WITHOUT_USER = "data/karla/karla-Teste_sem_usuario" # 3 x
INPUT_TEST_TR = "data/karla/karla-Teste_mao_esticada" # 3 x
INPUT_A = "data/karla/karla-A_perto_cotovelo" # x 3
INPUT_B = "data/karla/karla-B_perto_cotovelo" # x 3
INPUT_C = "data/karla/karla-C_perto_cotovelo" # x 3
INPUT_D = "data/karla/karla-D_perto_cotovelo" # x 3
INPUT_E = "data/karla/karla-E_perto_cotovelo" # x 3
OUTPUT = "data_segmented/karla"


def calculateTreshold():
    meanAveragedEmg = 0

    for turn in xrange(1,4):
        input_test_tr_complete = '%s-%d-emg.csv' % (INPUT_TEST_TR, turn)
        data = readCsv(input_test_tr_complete)
        averagedEmg = calculateAverageEnergy(data)
        print averagedEmg
        # print np.amax(averagedEmg)
        meanAveragedEmg += np.amax(averagedEmg)


    meanAveragedEmg = meanAveragedEmg / 3
    threshold = meanAveragedEmg * 0.05
    # print threshold
    return threshold

def segmentFiles():
    """
        1. Get data
        2. Segment Data by average emg
        3. Save this data in specific folder
    """

    threshold = calculateTreshold()

    for turn in xrange(1,4):
        input_a_complete = '%s-%d-emg.csv' % (INPUT_A, turn)
        input_b_complete = '%s-%d-emg.csv' % (INPUT_B, turn)
        input_c_complete = '%s-%d-emg.csv' % (INPUT_C, turn)
        input_d_complete = '%s-%d-emg.csv' % (INPUT_D, turn)
        input_e_complete = '%s-%d-emg.csv' % (INPUT_E, turn)

        # Get data
        data_a = readCsv(input_a_complete)
        data_b = readCsv(input_b_complete)
        data_c = readCsv(input_c_complete)
        data_d = readCsv(input_d_complete)
        data_e = readCsv(input_e_complete)



        # Segment Data by Average EMG
        averageEmgA = calculateAverageEnergy(data_a)
        averageEmgB = calculateAverageEnergy(data_b)
        averageEmgC = calculateAverageEnergy(data_c)
        averageEmgD = calculateAverageEnergy(data_d)
        averageEmgE = calculateAverageEnergy(data_e)

        if turn == 3:
            # print averageEmgA
            # printGraph(averageEmgA)
            pass

        segmentedA = segmentSignal(averageEmgA, threshold)
        segmentedB = segmentSignal(averageEmgB, threshold)
        segmentedC = segmentSignal(averageEmgC, threshold)
        segmentedD = segmentSignal(averageEmgD, threshold)
        segmentedE = segmentSignal(averageEmgE, threshold)

        # Save data in specific folder
        pathA = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "karla-A_perto_cotovelo", turn)
        pathB = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "karla-B_perto_cotovelo", turn)
        pathC = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "karla-C_perto_cotovelo", turn)
        pathD = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "karla-D_perto_cotovelo", turn)
        pathE = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "karla-E_perto_cotovelo", turn)


        writeDataInFile(segmentedA, pathA)
        writeDataInFile(segmentedB, pathB)
        writeDataInFile(segmentedC, pathC)
        writeDataInFile(segmentedD, pathD)
        writeDataInFile(segmentedE, pathE)


if __name__ == '__main__':
    # calculateTreshold()
    segmentFiles()
