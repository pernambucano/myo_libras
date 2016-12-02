#!/usr/local/bin/python
# -*- coding: utf-8 -*-


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, ShuffleSplit, cross_val_score, GridSearchCV, KFold, train_test_split,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import csv
from sklearn.pipeline import make_pipeline


# TODO: Put this method in a Util package
def writeDataInFile(data, output_filename):
    with open(output_filename, 'wb') as outputFile:
        writer = csv.writer(outputFile)
        for i in data:
            writer.writerow(i)


def classify(featureMatrix):
    n_estimators = 150 # This value changes to 80 when we are classifying 3 letters
    a,b = featureMatrix.shape
    X, y = featureMatrix[:, :b-1], featureMatrix[:, b-1]
    clf1 = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    print(np.mean(cross_val_score(clf1, X, y, cv=10)))

def plot_learning_curve(X, y):
    from sklearn.model_selection import learning_curve

    pipe = Pipeline([('clf', RandomForestClassifier(n_estimators=150, random_state=42))])

    train_sizes, train_scores, test_scores = \
                        learning_curve(estimator=pipe,
                                        X=X,
                                        y=y,
                                        train_sizes=np.linspace(0.1,1.0,10),
                                        cv=10,
                                        n_jobs=1,
                                        verbose=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Treinamento')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label=u'Validação')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Quantidade de dados de treinamento')
    plt.ylabel('Taxa de acerto')
    plt.legend(loc='lower right')
    plt.ylim([0, 1.2])
    plt.tight_layout()
    # plt.savefig('../figures/learning_curve.png', dpi=300)
    print "imprimindo imagem"
    plt.show()


def plot_n_trees(X, y):
    clf = RandomForestClassifier(random_state=42)
    param_range = [x for x in range(1,400,40)]
    train_scores, test_scores = validation_curve(
                    estimator=clf,
                    X=X,
                    y=y,
                    param_name='n_estimators',
                    param_range=param_range,
                    cv=10,
                    scoring = 'accuracy',
                    verbose=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='Treinamento')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label=u'Validação')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlim([0,400])
    plt.legend(loc='lower right')
    plt.xlabel(u'Número de árvores')
    plt.ylabel('Taxa de Acerto')
    plt.ylim([0, 1.2])
    plt.tight_layout()
    plt.show()