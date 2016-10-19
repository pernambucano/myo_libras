#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from segmentData import calculateAverageEnergy, segmentAveragedSignal, readCsv, writeDataInFile, segmentSensorsData
import numpy as np
import csv
from printGraph import printGraph
from calculateFeatures import getFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

user = "paulo"
INPUT_TEST_WITHOUT_USER = "data/%s/%s-Teste_sem_usuario" % (user, user)# 3 x
INPUT_TEST_TR = "data/%s/%s-Teste_mao_esticada"  % (user, user)# 3 x
INPUT_A = "data/%s/%s-A_perto_cotovelo" % (user, user) # x 3
INPUT_B = "data/%s/%s-B_perto_cotovelo"  % (user, user)# x 3
INPUT_C = "data/%s/%s-C_perto_cotovelo"  % (user, user)# x 3
INPUT_D = "data/%s/%s-D_perto_cotovelo"  % (user, user)# x 3
INPUT_E = "data/%s/%s-E_perto_cotovelo"  % (user, user)# x 3
OUTPUT = "data_segmented/%s" % user

INPUT_A_SEGMENTED = "data_segmented/%s/%s-A_perto_cotovelo"  % (user, user)
INPUT_B_SEGMENTED = "data_segmented/%s/%s-B_perto_cotovelo" % (user, user)
INPUT_C_SEGMENTED = "data_segmented/%s/%s-C_perto_cotovelo" % (user, user)
INPUT_D_SEGMENTED = "data_segmented/%s/%s-D_perto_cotovelo" % (user, user)
INPUT_E_SEGMENTED = "data_segmented/%s/%s-E_perto_cotovelo" % (user, user)


def readCsvSegmented(inputFileName):

    data = np.genfromtxt(inputFileName, delimiter=',', dtype=None)
    return data


def calculateTreshold():
    meanAveragedEmg = 0
    for turn in xrange(1,4):
        input_test_tr_complete = '%s-%d-emg.csv' % (INPUT_TEST_TR, turn)
        data = readCsv(input_test_tr_complete)
        averagedEmg = calculateAverageEnergy(data)
        # print averagedEmg
        # print np.amax(averagedEmg)
        meanAveragedEmg += np.amax(averagedEmg)


    meanAveragedEmg = meanAveragedEmg / 3
    threshold = meanAveragedEmg * 0.05
    # print threshold
    return threshold

def segmentFiles():

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


        startOfLetterA, endOfLetterA = segmentAveragedSignal(averageEmgA, threshold)
        startOfLetterB, endOfLetterB = segmentAveragedSignal(averageEmgB, threshold)
        startOfLetterC, endOfLetterC = segmentAveragedSignal(averageEmgC, threshold)
        startOfLetterD, endOfLetterD = segmentAveragedSignal(averageEmgD, threshold)
        startOfLetterE, endOfLetterE = segmentAveragedSignal(averageEmgE, threshold)



        data_a_segmented = segmentSensorsData(data_a, startOfLetterA, endOfLetterA)
        data_b_segmented = segmentSensorsData(data_b, startOfLetterB, endOfLetterB)
        data_c_segmented = segmentSensorsData(data_c, startOfLetterC, endOfLetterC)
        data_d_segmented = segmentSensorsData(data_d, startOfLetterD, endOfLetterD)
        data_e_segmented = segmentSensorsData(data_e, startOfLetterE, endOfLetterE)
        # printGraph(data_a_segmented)


        # # Save data in specific folder
        pathA = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "paulo-A_perto_cotovelo", turn)
        pathB = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "paulo-B_perto_cotovelo", turn)
        pathC = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "paulo-C_perto_cotovelo", turn)
        pathD = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "paulo-D_perto_cotovelo", turn)
        pathE = '%s/%s-%d-emg-segmented.csv' % (OUTPUT, "paulo-E_perto_cotovelo", turn)


        writeDataInFile(data_a_segmented, pathA)
        writeDataInFile(data_b_segmented, pathB)
        writeDataInFile(data_c_segmented, pathC)
        writeDataInFile(data_d_segmented, pathD)
        writeDataInFile(data_e_segmented, pathE)

def getAttributes():

    dataset = []
    for turn in xrange(1,4):
        input_a_complete_k = '%s-%d-emg-segmented.csv' % ("data_segmented/karla/karla-A_perto_cotovelo", turn)
        input_b_complete_k = '%s-%d-emg-segmented.csv' % ("data_segmented/karla/karla-B_perto_cotovelo", turn)
        input_c_complete_k = '%s-%d-emg-segmented.csv' % ("data_segmented/karla/karla-C_perto_cotovelo", turn)
        input_d_complete_k = '%s-%d-emg-segmented.csv' % ("data_segmented/karla/karla-D_perto_cotovelo", turn)
        input_e_complete_k = '%s-%d-emg-segmented.csv' % ("data_segmented/karla/karla-E_perto_cotovelo", turn)
        input_a_complete_p = '%s-%d-emg-segmented.csv' % ("data_segmented/paulo/paulo-A_perto_cotovelo", turn)
        input_b_complete_p = '%s-%d-emg-segmented.csv' % ("data_segmented/paulo/paulo-B_perto_cotovelo", turn)
        input_c_complete_p = '%s-%d-emg-segmented.csv' % ("data_segmented/paulo/paulo-C_perto_cotovelo", turn)
        input_d_complete_p = '%s-%d-emg-segmented.csv' % ("data_segmented/paulo/paulo-D_perto_cotovelo", turn)
        input_e_complete_p = '%s-%d-emg-segmented.csv' % ("data_segmented/paulo/paulo-E_perto_cotovelo", turn)

        # Get data
        data_a_k = readCsvSegmented(input_a_complete_k)
        data_b_k = readCsvSegmented(input_b_complete_k)
        data_c_k = readCsvSegmented(input_c_complete_k)
        data_d_k = readCsvSegmented(input_d_complete_k)
        data_e_k = readCsvSegmented(input_e_complete_k)
        data_a_p = readCsvSegmented(input_a_complete_p)
        data_b_p = readCsvSegmented(input_b_complete_p)
        data_c_p = readCsvSegmented(input_c_complete_p)
        data_d_p = readCsvSegmented(input_d_complete_p)
        data_e_p = readCsvSegmented(input_e_complete_p)

        # Get features
        dataset.append(getFeatures(data_a_k, 0).tolist())
        dataset.append(getFeatures(data_b_k, 1).tolist())
        dataset.append(getFeatures(data_c_k, 2).tolist())
        dataset.append(getFeatures(data_d_k, 3).tolist())
        dataset.append(getFeatures(data_e_k, 4).tolist())
        dataset.append(getFeatures(data_a_p, 0).tolist())
        dataset.append(getFeatures(data_b_p, 1).tolist())
        dataset.append(getFeatures(data_c_p, 2).tolist())
        dataset.append(getFeatures(data_d_p, 3).tolist())
        dataset.append(getFeatures(data_e_p, 4).tolist())

    ndataset =  np.array(dataset)
    # print ndataset[:,64]
    return ndataset

def feature_scaling(feature_matrix,target,reductor=None,scaler=None):
    lda = LDA(n_components=2)
    minmax = MinMaxScaler(feature_range=(-1,1))
    if not reductor:
        reductor = lda.fit(feature_matrix,target)
    feat_lda = reductor.transform(feature_matrix)
    if not scaler:
        scaler = minmax.fit(feat_lda)
    feat_lda_scaled = scaler.transform(feat_lda)

    return feat_lda_scaled,reductor,scaler

if __name__ == '__main__':
    segmentFiles()
    featureMatrix = getAttributes()

    [X,reductor,scaler] = feature_scaling(featureMatrix[:, :63], featureMatrix[:, 64])
    X_train, X_test, y_train, y_test = train_test_split(X, featureMatrix[:,64], test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(featureMatrix[:,:63], featureMatrix[:,64], test_size=0.2)

    # classifier = SVC(kernel='rbf', C=5, gamma=20)
    classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
    classifier.fit(X_train, y_train)

    # print classifier.predict(X_test[0])
    # print y_test[0]
    print classifier.score(X_test,y_test)
    # print "Classification accuracy for SVC is %0.2f" % (100 * classifier.score(X_test, y_test))

    # [X,reductor,scaler] = feature_scaling(featureMatrix[:, :63], featureMatrix[:, 64])
    #
    #
    # plt.scatter(X[0:30:5,0],X[0:30:5,1],c='r', marker='o', label='A')
    # plt.scatter(X[1:30:5,0],X[1:30:5,1],c='g', marker='>', label='B')
    # plt.scatter(X[2:30:5,0],X[2:30:5,1],c='c', marker='x', label='C')
    # plt.scatter(X[3:30:5,0],X[3:30:5,1],c='m', marker='+', label='D')
    # plt.scatter(X[4:30:5,0],X[4:30:5,1],c='b', marker='<', label='E')
    # plt.legend(scatterpoints=1,loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
