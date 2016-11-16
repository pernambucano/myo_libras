from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler



def feature_scaling(feature_matrix,target,reductor=None, scaler=None):
    lda = LDA(n_components=2)
    minmax = MinMaxScaler(feature_range=(-1,1))
    if not reductor:
        reductor = lda.fit(feature_matrix,target)
    feat_lda = reductor.transform(feature_matrix)
    if not scaler:
        scaler = minmax.fit(feat_lda)
    feat_lda_scaled = scaler.transform(feat_lda)

    return feat_lda_scaled,reductor,scaler

def classification(feature_matrix):
    label_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    X = feature_matrix[:, :64]
    y = feature_matrix[:, 64]
    # [X, reductor_tr, scaler_tr] = feature_scaling(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.20, random_state=42)

    lda = LDA(n_components=2)
    lda = lda.fit(X_train, y_train)
    X_train= lda.transform(X_train)

    # [X_train, reductor_tr, scaler_tr] = feature_scaling(X_train, y_train)

    print X_train[0]


    plt.scatter(X_train[:,0], X_train[:,1])
    plt.show()


    for label,marker,color in zip(
            np.unique(y),('^', 's', 'o', '>', '<'),('blue', 'red', 'green', 'black', 'magenta')):
        plt.scatter(X_train[y_train==label, 0], X_train[y_train==label, 1],
                    color=color, marker=marker)

    plt.show()




    X_test = lda.transform(X_test)

    # [X_test, reductor, scaler] = feature_scaling(X_test, y_test, reductor=reductor_tr, scaler=scaler_tr)


    print X_test[0]
    for label,marker,color in zip(
            np.unique(y),('^', 's', 'o', '>', '<'),('blue', 'red', 'green', 'black', 'magenta')):
        plt.scatter(X_test[y_test==label, 0], X_test[y_test==label, 1],
                    color=color, marker=marker)

    plt.show()

    clf = RandomForestClassifier(n_estimators=30, criterion="gini", max_depth=10)
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)
