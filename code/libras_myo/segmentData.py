import numpy as np


"""
TODO:
    * Receber dados (ok)
    * Dividir dado por sensor (ok)
    * Dividir em janelas (ok)
    * Calcular valores pra cada janela (ok)
    * Calcular valores de todas as janelas pra um sensor (ok)
    * Calcular valores para todos os sensores juntos (ok)
    * Definir o tamanho da janela
    * Salvar dados em arquivo
"""

WINDOW_SIZE = 128
WINDOW_STEP = 64
INPUT_FILENAME = "data/letra_a.csv"
OUTPUT_FILENAME = "data/letra_a_segmented.csv"
TR =  135 # twenty percent of the mean of the EW(t) of the signer’s maximal voluntary contraction.


## Calculate the beginning and the end of the LIBRAS signal

def calculateSegmentedSignal(segmentedData):
    pass

def calculateBeginningOfSignal(segmentedData):
    pass

def calculateEndingOfSignal(segmentedData):
    pass




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

def writeDataInFile(data):
    with open(OUTPUT_FILENAME, 'w') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerows(data)

# slidingWindow
# https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
def slidingWindow(sequence):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(WINDOW_SIZE) == type(0)) and (type(WINDOW_STEP) == type(0))):
        raise Exception("**ERROR** type(WINDOW_SIZE) and type(WINDOW_STEP) must be int.")
    if WINDOW_STEP > WINDOW_SIZE:
        raise Exception("**ERROR** WINDOW_STEP must not be larger than WINDOW_SIZE.")
    if WINDOW_SIZE > len(sequence):
        raise Exception("**ERROR** WINDOW_SIZE must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-WINDOW_SIZE)/WINDOW_STEP)+1

    # Do the work
    for i in range(0,numOfChunks*WINDOW_STEP,WINDOW_STEP):
        yield sequence[i:i+WINDOW_SIZE]


if __name__ == '__main__':
    data = readCsv()
    calculateValuesForAllSensors(data)
