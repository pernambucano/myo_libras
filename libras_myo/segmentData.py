#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt


WINDOW_SIZE = 16  # 80 ms
WINDOW_STEP = 8  # 50 ms
INPUT_FILENAME = "data/karla/karla-A_perto_cotovelo-1-emg.csv"
OUTPUT_FILENAME = "data_segmented/karla-A_perto_cotovelo-1-emg-segmented.csv"
OUTPUT_FILENAME_CUT = "data_segmented_cut/karla-A_perto_cotovelo-1-emg-segmented-cut.csv"



def segmentSensorsData(data, start, end):
    """
        Input: data from all sensors, start window, end window
        Output: sensors data segmented
    """
    startIndex = (start - 1) * WINDOW_STEP # Beginning of start window
    endIndex = end * (WINDOW_STEP + 1) # Ending of end window

    # print startIndex, endIndex
    result = []
    for sensorNumber in xrange(8):
        sensorData = getDataForSensor(sensorNumber, data)
        dataOfLetter = sensorData[startIndex:endIndex].tolist()
        result.append(dataOfLetter)

    return np.array(result).T.tolist()

# TR = twenty percent of the mean of the EW(t) of the signer’s maximal
# voluntary contraction.
def segmentAveragedSignal(averagedEmg, TR=135):
    """
        Input: average emg, threshold
        Output: index number of the first window delimiting the letter,
                index number of the last window delimiting the letter
    """

    data = averagedEmg
    listSize = len(data)

    # Put zeroes everytime the data is smaller than the threshold
    for item in xrange(0, listSize):
        if data[item] < TR:
            data[item] = 0

    # Get the indices for the nonzero sequences
    beforeCounter = 0
    afterCounter = 0

    mList = []
    while beforeCounter < listSize and afterCounter < listSize:
        start = 0
        end = 0

        if data[beforeCounter] != 0:
            start = beforeCounter

            afterCounter = beforeCounter
            while afterCounter < listSize and data[afterCounter] != 0:
                afterCounter += 1

            beforeCounter = afterCounter

            if afterCounter != 0:
                afterCounter -= 1

            end = afterCounter

            mList.append([start, end])

        else:
            beforeCounter += 1

    # get the sequence corresponding to the letter by getting the biggest Difference
    startOfLetter = 0
    endOfLetter = 0
    biggestDifference = 0

    for start, end in mList:
        if (end - start) > biggestDifference:
            biggestDifference = end - start
            startOfLetter = start
            endOfLetter = end

    # writeDataInFile(data[startLetter:endLetter + 1], OUTPUT_FILENAME_CUT)
    # return data[startOfLetter:endOfLetter + 1]
    return startOfLetter, endOfLetter

# Calculate the Average Energy of the 8 Sensors
def calculateAverageEnergy(data):
    """
        Get data from csv file and return a vector of N values. Each value correspond to a window.
        This value is the result of equation 1 for each window.
    """

    allSensorsWindowsValues = []
    for sensorNumber in range(8):
        sensorData = getDataForSensor(sensorNumber, data)
        arrayOfWindows = transformDataIntoWindows(sensorData)
        arrayOfWindowValues = calculateAndSaveWindowValuesOfSensorInArray(
            arrayOfWindows)
        allSensorsWindowsValues.append(arrayOfWindowValues)

    # make a numpy array to transpose it
    sensorsWindowsValuesArrayTransposed = np.transpose(
        np.array(allSensorsWindowsValues))
    sensorsWindowsValuesArraySumed = np.sum(
        sensorsWindowsValuesArrayTransposed, axis=1)
    sensorsWindowsValuesArrayResult = map(
        (lambda y: y / 8), sensorsWindowsValuesArraySumed)


    # writeDataInFile(sensorsWindowsValuesArrayResult)
    return sensorsWindowsValuesArrayResult


def transformDataIntoWindows(sensorData):
    windowsArray = []
    windows = slidingWindow(sensorData)

    for window in windows:
        window = window.tolist()
        windowsArray.append(window)
    return windowsArray


def calculateValueOfWindow(window):
    valueWindow = 0
    counter = 1

    for x in window:
        valueWindow += pow(x, 2) # maybe it would be useful to sum
                                 # before use the exponential
        counter += 1

    return (valueWindow / counter)


def calculateAndSaveWindowValuesOfSensorInArray(windows):
    dataWindowsSensorArray = []
    for window in windows:
        valueOfWindow = calculateValueOfWindow(window)
        # save valueOfWindow in array
        dataWindowsSensorArray.append(valueOfWindow)
    return dataWindowsSensorArray


def readCsv(inputFileName=INPUT_FILENAME):

    data = np.genfromtxt(inputFileName, delimiter=',', dtype=None)
    data = data[:, 1:]  # We are not using the timestamp at this time
    return data


def getDataForSensor(sensorNumber, dataMatrix):
    return dataMatrix[:, sensorNumber]


def writeDataInFile(data, output_filename=OUTPUT_FILENAME):
    with open(output_filename, 'wb') as outputFile:
        writer = csv.writer(outputFile)
        for i in data:
            writer.writerow(i)


# slidingWindow
# https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
def slidingWindow(sequence):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    window_size = WINDOW_SIZE
    window_step = WINDOW_STEP

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(window_size) == type(0)) and (type(window_step) == type(0))):
        raise Exception(
            "**ERROR** type(WINDOW_SIZE) and type(WINDOW_STEP) must be int.")
    if window_step > window_size:
        raise Exception(
            "**ERROR** WINDOW_STEP must not be larger than WINDOW_SIZE.")
    if window_size > len(sequence):
        raise Exception(
            "**ERROR** WINDOW_SIZE must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - window_size) / window_step) + 1

    # Do the work
    for i in range(0, numOfChunks * window_step, window_step):
        yield sequence[i:i + window_size]


def printGraphTest(sequence):
    # plt.xticks([x for x in xrange(0, 27)])
    plt.grid(True)
    plt.title('Letra segmentada')
    plt.xlabel('windows')
    plt.ylabel('energy')
    plt.plot(sequence)
    plt.savefig("teste.png")

if __name__ == '__main__':
    data = readCsv()
    results = calculateAverageEnergy(data)
    segmented = segmentSignal(results)
    # printGraphTest(segmented)
