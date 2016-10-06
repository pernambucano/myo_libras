#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt


"""
TODO:
    * Receber dados (ok)
    * Dividir dado por sensor (ok)
    * Dividir em janelas (ok)
    * Calcular valores pra cada janela (ok)
    * Calcular valores de todas as janelas pra um sensor (ok)
    * Calcular valores para todos os sensores juntos (ok)
    * Salvar dados em arquivo (ok)
    * Definir o tamanho da janela
    * Definir se é necessário "te - ts + 1 > 8"
    * Colar te e ts ao TR (ok)
    * Criar código pra printar os grafos
    * Criar codigo pra pegar os dados mais facilmente
    * Calcular proporção segundo o paper
"""
#
WINDOW_SIZE = 10 # TODO: Definir
WINDOW_STEP = 5 # TODO: Definir
INPUT_FILENAME = "data/letra_a.csv"
OUTPUT_FILENAME = "data/letra_a_segmented.csv"
OUTPUT_FILENAME_CUT = "data/letra_a_segmented_cut.csv"
TR =  135 # twenty percent of the mean of the EW(t) of the signer’s maximal voluntary contraction.


## Calculate the beginning and the end of the LIBRAS signal
## Testar possibilidade de "te - ts + 1 > N"
startWindow = 0
def calculateSegmentedSignal(segmentedData):
    start = -1
    end = -1
    global startWindow

    windows = slidingWindow(segmentedData, cutWindow=True)
    for window in windows:
        start = calculateBeginningOfSignal(window, start)
        end = calculateEndingOfSignal(window, end)
        startWindow += 1

    # print start, end
    result = addTRBeforeAndAfter(segmentedData[start:end+1])
    # writeDataInFile(result, OUTPUT_FILENAME_CUT)
    return result

def addTRBeforeAndAfter(sequence):
    localSequence = sequence
    localSequence.insert(0, TR)
    localSequence.append(TR)
    return localSequence

def calculateBeginningOfSignal(window, currentValue):
    global startWindow

    if window[0] < TR and window[1] < TR and window[2] > TR and window[3] > TR and window[4] > TR:
        return (startWindow + 2) # Sera que window[0] seria melhor?
    else:
        return currentValue

def calculateEndingOfSignal(window, currentValue):
    global startWindow

    if window[0] > TR and window[1] > TR and window[2] > TR and window[3] < TR and window[4] < TR:
        return (startWindow + 2) # Sera que window[4] seria melhor?
    else:
        return currentValue


## Calculate the Average Energy of the 8 Sensors
def calculateValuesForAllSensors(data):
    """
        Get data from csv file and return a vector of N values. Each value correspond to a window.
        This value is the result of equation 1 for each window.
    """
    allSensorsWindowsValues = []
    for sensorNumber in range(8):
        sensorData = getDataForSensor(sensorNumber, data)
        arrayOfWindows = transformSensorDataIntoArrayOfWindows(sensorData)
        arrayOfWindowValues = calculateAndSaveWindowValuesOfSensorInArray(arrayOfWindows)
        allSensorsWindowsValues.append(arrayOfWindowValues)

    # make a numpy array to transpose it
    sensorsWindowsValuesArrayTransposed = np.transpose(np.array(allSensorsWindowsValues))
    sensorsWindowsValuesArraySum =  np.sum(sensorsWindowsValuesArrayTransposed, axis=1)
    sensorsWindowsValuesArrayResult = map((lambda y: y/8), sensorsWindowsValuesArraySum)

    # Change this to save in a csv file
    print sensorsWindowsValuesArrayResult
    writeDataInFile(sensorsWindowsValuesArrayResult)
    return sensorsWindowsValuesArrayResult

def transformSensorDataIntoArrayOfWindows(sensorData):
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
        valueWindow += pow(x, 2)
        counter += 1

    return (valueWindow / counter)


def calculateAndSaveWindowValuesOfSensorInArray(windows):
    dataWindowsSensorArray = []
    for window in windows:
        valueOfWindow = calculateValueOfWindow(window)

        # save valueOfWindow in array
        # dataWindowsSensorArray = np.append(dataWindowsSensorArray, valueOfWindow)
        dataWindowsSensorArray.append(valueOfWindow)
    return dataWindowsSensorArray

def readCsv():
    data = np.genfromtxt(INPUT_FILENAME, delimiter=',', dtype=None)
    return data

def getDataForSensor(sensorNumber, dataMatrix):
    return dataMatrix[:,sensorNumber]

def writeDataInFile(data, output_filename=OUTPUT_FILENAME):
    with open(output_filename, 'wb') as outputFile:
        writer = csv.writer(outputFile)
        for i in data:
            writer.writerow([i])


# slidingWindow
# https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
def slidingWindow(sequence, cutWindow=False):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""


    if cutWindow:
        window_size = 5
        window_step = 1
    else:
        window_size = WINDOW_SIZE
        window_step = WINDOW_STEP

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(window_size) == type(0)) and (type(window_step) == type(0))):
        raise Exception("**ERROR** type(WINDOW_SIZE) and type(WINDOW_STEP) must be int.")
    if window_step > window_size:
        raise Exception("**ERROR** WINDOW_STEP must not be larger than WINDOW_SIZE.")
    if window_size > len(sequence):
        raise Exception("**ERROR** WINDOW_SIZE must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-window_size)/window_step)+1

        # Do the work
    for i in range(0,numOfChunks*window_step,window_step):
        yield sequence[i:i+window_size]

def printGraphTest(sequence):
    plt.xticks([x for x in xrange(0,27)])
    plt.grid(True)
    plt.title('Letra A Segmentada')
    plt.xlabel('windows')
    plt.ylabel('energy')
    plt.plot(sequence)
    plt.savefig("teste.png")

if __name__ == '__main__':
    data = readCsv()
    results = calculateValuesForAllSensors(data)
    printGraphTest(calculateSegmentedSignal(results))
