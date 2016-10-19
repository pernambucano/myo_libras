import numpy as np
import nitime as nt
from segmentData import getDataForSensor

def readCsv(inputFileName):

    data = np.genfromtxt(inputFileName, delimiter=',', dtype=None)
    return data

def mav(segment):
    mav = np.mean(np.abs(segment))
    return mav


def var(segment):
    var = np.var(segment)
    return var


def rms(segment):
    rms = np.sqrt(np.mean(np.power(segment, 2)))
    return rms


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


def getFeatures(data, letterClass):

    listOfFeatures = [mav, var, rms, zc, arc]
    features = np.zeros(65)  # 64 attributes  + 1 class
    featureData = []
    n = 0


    for channel in range(8):
        # print channel
        sensorData = getDataForSensor(channel, data)
        for feature in listOfFeatures:
            # print feature
            if feature == arc:
                # print 'it\'s arc'
                c1, c2, c3, c4 = feature(sensorData)
                features[n] = c1
                features[n + 1] = c2
                features[n + 2] = c3
                features[n + 3] = c4
                n += 4
            else:
                features[n] = feature(sensorData)
                n += 1
        # print features
        # featureData.append(features)

    features[64] = letterClass # y

    # print features
    return features

if __name__ == '__main__':
    data = readCsv('data_segmented/karla/karla-A_perto_cotovelo-1-emg-segmented.csv')
    getFeatures(data, 1)
