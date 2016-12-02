#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier


def plotSensores():
    ### Letra A
    df_a = pd.read_csv("data/bernardo/bernardo-A-3-emg.csv")

    df_a_avg = np.genfromtxt("data/bernardo/bernardo-A-3-avg.csv", delimiter=',', dtype=float)
    df_a_avg = pd.DataFrame(df_a_avg)
    df_a = df_a.drop(df_a.columns[[0]], axis=1)
    df_a.columns = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6", "sensor7", "sensor8"]

    ### Letra B
    df_b = pd.read_csv("data/bernardo/bernardo-D-3-emg.csv")

    df_b_avg = np.genfromtxt("data/bernardo/bernardo-D-3-avg.csv", delimiter=',', dtype=float)
    df_b_avg = pd.DataFrame(df_b_avg)
    df_b = df_b.drop(df_b.columns[[0]], axis=1)
    df_b.columns = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6", "sensor7", "sensor8"]



    plt.figure()
    fig, axes = plt.subplots(figsize=(8,8),nrows=2, ncols=2)

    ## Letra A
    df_a.plot(legend=False,ax=axes[0, 0])
    axes[0,0].set_ylabel("Letra A - Todos os sensores")
    df_a_avg.plot(legend=False,ax=axes[1, 0])
    axes[1,0].set_ylabel(u"Letra A - Valores Médios");


    ## Letra A
    df_b.plot(legend=False,ax=axes[0, 1])
    axes[0,1].set_ylabel("Letra D - Todos os sensores")
    df_b_avg.plot(legend=False,ax=axes[1, 1])
    axes[1,1].set_ylabel(u"Letra D - Valores Médios");


    for ax in axes:
        for bx in ax:
            bx.set_xticks([], [])
            bx.set_yticks([], [])

    plt.show()

def plot_crossvalidation():

    plt.figure()
    df = pd.Series([78,78,76,62], index=["Grupo 1", "Grupo 2", "Grupo 3", "Grupo 4"])
    ax =df.plot(kind='bar', rot=0, title="10-Fold Cross-Validation")
    ax.grid(True, which='major', axis='y')
    ax.set_ylim(0,100)
    plt.show()

def plotLeaveOneOut():
    featureMatrix = pd.read_csv("featureMatrix.csv")
    n_estimators = 150
    i = 5

    ## 15 - 0.333333333333 0.4 0.466666666667 0.733333333333 0.733333333333
    ## 10 - 0.5 0.5 0.6 0.6 0.6
    ## 5  - 0.8 0.6 0.4 1.0 0.8
    ## 3  - 0.666666666667 0.666666666667 0.666666666667 1.0 1.0
    #
    # s1 = featureMatrix.iloc[0:i,:]
    # s2 = featureMatrix.iloc[i:i*2,:]
    # s3 = featureMatrix.iloc[i*2:i*3,:]
    # s4 = featureMatrix.iloc[i*3:i*4,:]
    # s5 = featureMatrix.iloc[i*4:i*5,:]
    #
    # ### W/o S1
    # trainingwos1 = s2.append(s3).append(s4).append(s5)
    # clf = RandomForestClassifier(n_estimators=n_estimators, random_state=30)
    # clf.fit(trainingwos1.iloc[:,:24], trainingwos1.iloc[:,24])
    # scorewos1 = clf.score(s1.iloc[:,:24], s1.iloc[:,24])
    # ### W/o S2
    # trainingwos2 = s1.append(s3).append(s4).append(s5)
    # clf = RandomForestClassifier(n_estimators=n_estimators, random_state=30)
    # clf.fit(trainingwos2.iloc[:,:24], trainingwos2.iloc[:,24])
    # scorewos2 = clf.score(s2.iloc[:,:24], s2.iloc[:,24])
    # ### W/o S3
    # trainingwos3 = s1.append(s2).append(s4).append(s5)
    # clf = RandomForestClassifier(n_estimators=n_estimators, random_state=30)
    # clf.fit(trainingwos3.iloc[:,:24], trainingwos3.iloc[:,24])
    # scorewos3 = clf.score(s3.iloc[:,:24], s3.iloc[:,24])
    # ### W/o S4
    # trainingwos4 = s1.append(s2).append(s3).append(s5)
    # clf = RandomForestClassifier(n_estimators=n_estimators, random_state=30)
    # clf.fit(trainingwos4.iloc[:,:24], trainingwos4.iloc[:,24])
    # scorewos4 = clf.score(s4.iloc[:,:24], s4.iloc[:,24])
    # ### W/o S5
    # trainingwos5 = s1.append(s2).append(s3).append(s4)
    # clf = RandomForestClassifier(n_estimators=n_estimators, random_state=30)
    # clf.fit(trainingwos5.iloc[:,:24], trainingwos5.iloc[:,24])
    # scorewos5 = clf.score(s5.iloc[:,:24], s5.iloc[:,24])
    # print scorewos1, scorewos2, scorewos3, scorewos4, scorewos5

    plt.figure()
    mdict = {'Grupo 1': [0.66, 0.66, 0.66, 1.0, 1.0], 'Grupo 2': [0.8, 0.6, 0.4, 1.0, 0.8],
                        'Grupo 3':[0.5, 0.5, 0.6, 0.6, 0.6], 'Grupo 4': [0.33, 0.4, 0.46, 0.73, 0.73]}
    df = pd.DataFrame(mdict)
    df.index = ["P1", "P2", "P3", "P4", "P5"]
    ax = df.plot(kind='bar', title=u"Validação por 'Leave One Subject Out'", rot=0)
    ax.set_ylim(0,1.2)
    ax.grid(True, which='major', axis='y')
    plt.show()