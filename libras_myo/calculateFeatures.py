import numpy as np
import nitime as nt

def mav(segment):
    mav = np.mean(np.abs(segment))
    return mav

def var(segment):
    var = np.var(segment)
    return var

def rms(segment):
    rms = np.sqrt(np.mean(np.power(segment,2)))
    return rms

def zc(segment):
    nz_segment = []
    nz_indices = np.nonzero(segment)[0]
    for m in nz_indices:
        nz_segment.append(segment[m])
    N = len(nz_segment)
    zc = 0
    for n in range(N-1):
        if((nz_segment[n]*nz_segment[n+1]<0) and np.abs(nz_segment[n]-nz_segment[n+1]) >= 1e-4):
            zc = zc + 1
    return zc

def arc(segment):
    cf, var = nt.algorithms.autoregressive.AR_est_LD(a,4)
    return cf
