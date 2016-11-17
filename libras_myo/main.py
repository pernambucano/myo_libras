#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from segmentData import calculateAverageEnergy, segmentAveragedSignal, readCsv, writeDataInFile, segmentSensorsData, segmentContinuousData, segmentAccData
import numpy as np
import csv
from calculateFeatures import getFeaturesEmg, getFeaturesAcc, getFeatures
import matplotlib.pyplot as plt
from classification import classify
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
import itertools

# listOfLetters = ['A', 'B', 'C' , 'D', 'E', 'F' , 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'U', 'V']
# listOfLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N','O', 'P', 'Q', 'R']
# listOfLetters = ['A', 'B', 'C', 'D', 'E', 'G', 'I', 'L', 'M', 'O']
# listOfLetters = ['A', 'B', 'C', 'D', 'E'] #
listOfLetters = ['A', 'B', 'C'] # n_estimators = 2, max_depth = 10


users = ['yanna', 'antonio', 'bernardo', 'roniero', 'karla']

directory = "data"
directorySegmented = "data_segmented"

def readCsvSegmented(inputFileName):

    data = np.genfromtxt(inputFileName, delimiter=',')
    return data

def calculateTreshold(user):
    meanAveragedEmg = 0
    if user != 'johnathan':
        for turn in xrange(1, 4):
            path = "%s/%s/%s-Mao_esticada-%s-emg.csv" % (directory, user, user, turn)
            data = readCsv(path)
            averagedEmg = calculateAverageEnergy(data)
            meanAveragedEmg += np.amax(averagedEmg)

        meanAveragedEmg = meanAveragedEmg / 3
        threshold = meanAveragedEmg * 0.05
        return threshold
    elif user == 'johnathan':
        return 0

def segmentFile(path, user):
    threshold = calculateTreshold(user)
    data = readCsv(path)
    averagedEmg = calculateAverageEnergy(data)
    letterIndexes = segmentContinuousData(averagedEmg)
    print letterIndexes
    print len(letterIndexes)



def segmentFiles():
    for user in users:
        threshold = calculateTreshold(user) # Calcula o  threshold de cada usuário
        for turn in xrange(1, 4):
            for letter in listOfLetters:
                pathEmg = "%s/%s/%s-%s-%s-emg.csv" % (directory, user, user, letter, turn)
                pathAcc = "%s/%s/%s-%s-%s-acc.csv" % (directory, user, user, letter, turn)
                # print path

                # Get data
                dataEmg = readCsv(pathEmg)

                # dataAcc = readCsv(pathAcc)

                # Segment Data by Average EMG
                averageEmg = calculateAverageEnergy(dataEmg)

                startOfLetter, endOfLetter = segmentAveragedSignal(
                averageEmg, threshold)

                # Segment ACC Data
                # data_segmented_acc = segmentAccData(dataAcc, startOfLetter, endOfLetter)

                # Segment EMG Data
                data_segmented_emg = segmentSensorsData(
                dataEmg, startOfLetter, endOfLetter)


                # # Save data in specific folder
                path_segmented_emg = "%s/%s/%s-%s-%s-emg-segmented.csv" % (directorySegmented, user, user, letter, turn)
                # path_segmented_acc = "%s/%s/%s-%s-%s-acc-segmented.csv" % (directorySegmented, user, user, letter, turn)

                writeDataInFile(data_segmented_emg, path_segmented_emg)
                # writeDataInFile(data_segmented_acc, path_segmented_acc)


def getAttributes(listOfLetters):


    dataset = []

    for user in users:
        for turn in xrange(1, 4):
                classLetter = 1
                for letter in listOfLetters:

                    path_segmented_emg = "%s/%s/%s-%s-%s-emg-segmented.csv" % (directorySegmented, user, user, letter, turn)
                    # path_segmented_acc = "%s/%s/%s-%s-%s-acc-segmented.csv" % (directorySegmented, user, user, letter, turn)

                    # Get data
                    data_emg = readCsvSegmented(path_segmented_emg)
                    # data_acc = readCsvSegmented(path_segmented_acc)



                    # Get features
                    # dataset.append(getFeaturesEmg(data_emg, classLetter))
                    # dataset.append(getFeaturesAcc(data_acc, classLetter))

                    dataset.append(getFeatures(data_emg, classLetter))

                    classLetter += 1



    ndataset = np.array(dataset)
    return ndataset

if __name__ == '__main__':
    # segmentFiles()
    featureMatrix = getAttributes(listOfLetters)
    # writeDataInFile(featureMatrix, 'featureMatrix.csv')
    classify(featureMatrix)
    print "----------------"
#
