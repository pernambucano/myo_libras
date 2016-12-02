import numpy as np
import nitime as nt
from segmentData import getDataForSensor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from util import entropy
from scipy.stats import kurtosis


# Put this function in a util.py file
def readCsv(inputFileName):
    data = np.genfromtxt(inputFileName, delimiter=',', dtype=None)
    return data


## Many of these feature functions were taken from 
# https://github.com/RamanSinghca/MultiChannelMYOGestureClassification/blob/master/EMGFeatureExtraction.py


# Waveform Length
def waveformLength(x):
    tempVal=0
    N= len(x)
    for i in range(N-1):
        currentEle=x[i]
        nextEle=x[i+1]
        tempVal=tempVal+np.absolute(nextEle-currentEle)
    finalVal=tempVal
    return finalVal

#Willison Amplitude
def wamp(x, threshold=30):
    tempVal=0
    N= len(x)
    for i in range(N-1):
        currentEle=x[i]
        nextEle=x[i+1]
        fval=currentEle-nextEle
        if np.absolute(fval)>=threshold:
            tempVal=tempVal+1
    finalVal=tempVal
    return finalVal

#Energy of a time series Ei
def ssi(segment):
    '''
    Energy of a time series Ei
    '''
    return np.dot(segment, segment)

# Integrated EMG
def iemg(x):
    tempVal=0
    for i in x:
        tempVal=tempVal+np.absolute(i)
    return tempVal


def mav(segment):
    mav = np.mean(np.abs(segment))
    return mav

def mean(segment):
    mean = np.mean(segment)
    return mean


def var(segment):
    var = np.var(segment)
    return var


def rms(segment):
    rms = np.sqrt(np.mean(np.power(segment, 2)))
    return rms

def std(segment):
    std = np.std(segment)
    return std

#Modified Mean Absolute Value1
def mav1(x):
    tempVal=0
    N= len(x)
    for i in x:
        if 0.25*N<=i<=0.75*N:
            tempVal=tempVal+np.absolute(i)
        else:
            tempVal=tempVal+0.5*np.absolute(i)

    finalMAV=tempVal/N
    return finalMAV

#Modified Mean Absolute Value2
def mav2(x):
    tempVal=0
    N= len(x)
    for i in x:
        if 0.25*N<=i<=0.75*N:
            tempVal=tempVal+np.absolute(i)
        elif i<0.25*N:
            tempVal=tempVal+(4*np.absolute(i))/N
        else:
            tempVal=tempVal+(4*(np.absolute(i)-N))/N


    finalMAV=tempVal/N
    return finalMAV


def zc(segment):
    nz_segment = []
    nz_indices = np.nonzero(segment)[0]
    for m in nz_indices:
        nz_segment.append(segment[m])
    N = len(nz_segment)
    zc = 0
    for n in range(N - 1):
        if((nz_segment[n] * nz_segment[n + 1] < 0) and np.abs(nz_segment[n] - nz_segment[n + 1]) >= 1e-4):
            zc = zc + 1
    return zc


def arc(segment):
    cf, var = nt.algorithms.autoregressive.AR_est_LD(segment, 4)
    return cf

def getFeatures(dataEmg, letterClass):
    featuresEmg = getFeaturesEmg(dataEmg)
    features = np.hstack((featuresEmg, letterClass))
    return features

#Slope Sign change
def ssc(x, threshold=30):
    tempVal=0
    N= len(x)
    for i in range(1,(N-1)):
        previousEle=x[i-1]
        currentEle=x[i]
        nextEle=x[i+1]
        fval=((currentEle-previousEle)*(currentEle-nextEle))
        if fval>=threshold:
            tempVal=tempVal+1
    finalVal=tempVal
    return finalVal


def getFeaturesEmg(data):
    nAttributes = 3
    listOfFeatures = [mav, var, wamp]
    features = np.zeros(nAttributes*8)  
    featureData = []
    n = 0

    allChannels = []
    for channel in range(8):
        allChannels.append(channel)
        sensorData = getDataForSensor(channel, data)
        for feature in listOfFeatures:
            if feature == arc:
                c1, c2, c3, c4 = feature(sensorData)
                features[n] = c1
                features[n + 1] = c2
                features[n + 2] = c3
                features[n + 3] = c4
                n += 4
            else:
                features[n] = feature(sensorData)
                n+=1
    return features

def getFeaturesAcc(data):
    nAtts = 4
    listOfFeatures = [mav, mean, var, ssi]

    features = np.zeros(nAtts*3)
    featureData = []
    n = 0

    allChannels = []
    for channel in range(3):
        sensorData = getDataForSensor(channel, data)
        for feature in listOfFeatures:
            if feature == arc:
                c1, c2, c3, c4 = feature(sensorData)
                features[n] = c1
                features[n + 1] = c2
                features[n + 2] = c3
                features[n + 3] = c4
                n += 4
            else:
                features[n] = feature(sensorData)
                n+=1
    return features


if __name__ == '__main__':
    data = readCsv('data_segmented/karla/karla-A_perto_cotovelo-1-emg-segmented.csv')
    getFeatures(data, 1)
