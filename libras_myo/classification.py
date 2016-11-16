from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import validation_curve, ShuffleSplit, cross_val_score, GridSearchCV, KFold, train_test_split,StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from printGraph import printGraph, print3D
from sklearn.decomposition import PCA
from utils import visualize_tree
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from minisom import MiniSom
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from plot_classifier_comparison import printScores
from collections import Counter
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def writeDataInFile(data, output_filename):
    with open(output_filename, 'wb') as outputFile:
        writer = csv.writer(outputFile)
        for i in data:
            writer.writerow(i)



def classify(featureMatrix):
    a,b = featureMatrix.shape
    X, y = featureMatrix[:, :b-1], featureMatrix[:, b-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)


    # pca = PCA(n_components=2)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    #
    # plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.show()

    # clf = RandomForestClassifier(n_estimators=250, max_depth=100)
    # clf.fit(X_train,y_train)
    # print clf.score(X_test, y_test)

    # pipe_lr = Pipeline([('clf', RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42))])
    # #
    # scores = cross_val_score(estimator=pipe_lr,
    #                      X=X,
    #                      y=y,
    #                      cv=10,
    #                      n_jobs=1,
    #                      verbose=1)
    #
    # print('CV accuracy scores: %s' % scores)
    # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # pipe_lr.fit(X_train, y_train)
    # print pipe_lr.score(X_test, y_test)


    # printGraph(X_train, y_train)
    # diagnose_bias(X_train,y_train)
    # fitting(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=350, max_depth=40, random_state=42)

    print(np.mean(cross_val_score(clf, X, y, cv=10)))

    # param_range_estimators = [x for x in range(10,400,40)]
    # param_range_max_depth = [x for x in range(1,50,10)]
    # param_range_criterion = ['gini', 'entropy']
    # param_bool = [True, False]
    # param_weight = ['balanced', None]
    #
    # param_grid = [{
    #     'n_estimators': param_range_estimators,
    #     'max_depth': param_range_max_depth,
    #     'criterion': param_range_criterion,
    #     'random_state': [42]}]
    #
    # gs = GridSearchCV(estimator=clf,
    #                 param_grid=param_grid,
    #                 scoring='accuracy',
    #                 cv = 10,
    #                 verbose=10)
    # gs = gs.fit(X_train, y_train)
    # print(gs.best_score_)
    # print(gs.best_params_)

def diagnose_bias(X_train, y_train):
    from sklearn.model_selection import learning_curve

    pipe = Pipeline([('clf', RandomForestClassifier(n_estimators=250, max_depth=100))])

    train_sizes, train_scores, test_scores = \
                        learning_curve(estimator=pipe,
                                        X=X_train,
                                        y=y_train,
                                        train_sizes=np.linspace(0.1,1.0,10),
                                        cv=5,
                                        n_jobs=1,
                                        verbose=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0, 1.0])
    plt.tight_layout()
    # plt.savefig('./figures/learning_curve.png', dpi=300)
    print "imprimindo imagem"
    plt.show()


def fitting(X_train, y_train):
    clf = RandomForestClassifier(random_state=42, max_depth=10)
    param_range = [x for x in range(1,100,10)]
    train_scores, test_scores = validation_curve(
                    estimator=clf,
                    X=X_train,
                    y=y_train,
                    param_name='n_estimators',
                    param_range=param_range,
                    cv=5,
                    scoring = 'accuracy',
                    verbose=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlim([0,400])
    plt.legend(loc='lower right')
    plt.xlabel('# of Trees')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])
    plt.tight_layout()
    # plt.savefig('./figures/validation_curve.png', dpi=300)
    plt.show()


#     # listSVMs = train_worst_pairs(X_train, y_train)
#     # impossibleClassesTrain = impossibleClasses(X_train, listSVMs)
#     # impossibleClassesTrain = list(set(impossibleClassesTrain))
#     # print impossibleClasape
#     # for ict in impossibleClassesTrain:
#     #     indices_train = [np.where(y_train == ict)]
#     #     y_train = np.delete(y_train, indices_train, axis=0)
#     #     X_train = np.delete(X_train, indices_train, axis=0)
#
#     # scaller = StandardScaler()
#     # X_train = scaller.fit_transform(X_train, y_train)
#     # X_test = scaller.transform(X_test)
#
#     # ovr = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
#     # clf = RandomForestClassifier(n_estimators=400, max_depth=40)
#     # classifyTest(X_train, y_train)
#
#
#     # clf.fit(X_train, y_train)
#     # print clf.score(X_test, y_test)
#     #
#     # scores =  cross_val_score(clf, X,y, cv=10)
#     # print scores
#     # print scores.mean()
#
# #     # printScores(X,y)
# #     """
# #
# #         1. Divide by train/test datasets
# #         2. Train worsts pairs in svm
# #         3. During train, train svms on worst pairs
# #         4. During test, cut off those which are impossible
# #         X. Do this in a 10-fold cross validation
# #
# #     """
# #
#     # X, y = featureMatrix[:, :64], featureMatrix[:, 64]
#     # kf = KFold(n_splits=10, shuffle=True)
#     # score = 0
#     # for train_index, test_index in kf.split(X):
#     #     X_train, X_test = X[train_index], X[test_index]
#     #     y_train, y_test = y[train_index], y[test_index]
#     #
#     #     # scaller = StandardScaler()
#     #     # X_train = scaller.fit_transform(X_train, y_train)
#     #     # X_test =scaller.transform(X_test)
#     #
#     #
#     #     # listSVMs = train_worst_pairs(X_train, y_train)
#     #     #
#     #     # impossibleClassesTrain = impossibleClasses(X_train, listSVMs)
#     #     # impossibleClassesTest = impossibleClasses(X_test, listSVMs)
#     #     #
#     #     # impossibleClassesTrain = list(set(impossibleClassesTrain))
#     #     # print impossibleClassesTrain
#     #     # print X_train.shape
#     #     #
#     #     # # for i in impossibleClassesTrain:
#     #     # #     X_train =  X_train[y_train != i]
#     #     #     print "1. %s" %  y_train
#     #     #     y_train = y_train[y_train != i]
#     #     #     print "2. %s" %  y_train
#     #     #
#     #     # # writeDataInFile(X_train, "xtrain.csv")
#     #     # print y_train
#     #     # print y_test
#     #
#     #     # Remover de y_test e y_train os valores impossiveis
#     #
#     #     rf = RandomForestClassifier(n_estimators=300, max_depth=40, random_state=30)
#     #     rf.fit(X_train, y_train)
#     #     score +=  rf.score(X_test, y_test)
#     #
#     # score = score / 10
#     # print score
#
#
# def classifyTest(X_train, y_train):
#     from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
#
#     import numpy as np
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.pipeline import Pipeline
#
#     from sklearn.model_selection import cross_val_score
#
#     clf1 = LogisticRegression(penalty='l2',
#                               C=0.001,
#                               random_state=0)
#
#     clf2 = DecisionTreeClassifier(max_depth=30,
#                                   criterion='entropy',
#                                   random_state=0)
#
#     clf3 = KNeighborsClassifier(n_neighbors=1,
#                                 p=2,
#                                 metric='minkowski')
#
#     clf4 = RandomForestClassifier(n_estimators=400, max_depth=40,random_state=42)
#
#     pipe1 = Pipeline([['sc', StandardScaler()],
#                       ['clf', clf1]])
#     pipe3 = Pipeline([['sc', StandardScaler()],
#                       ['clf', clf3]])
#
#     clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN', 'RF']
#
#     print('10-fold cross validation:\n')
#     for clf, label in zip([pipe1, clf2, pipe3, clf4], clf_labels):
#         scores = cross_val_score(estimator=clf,
#                                  X=X_train,
#                                  y=y_train,
#                                  cv=10)
#         print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
#               % (scores.mean(), scores.std(), label))
#
# def impossibleClasses(X, listSVMs):
#     # print "Stuff"
#     impossibleClassesList = []
#     for clf in listSVMs:
#         samples = clf.predict(X)
#         [(least_common, n)] = Counter(samples).most_common()[:-1-1:-1]
#         impossibleClassesList.append(least_common)
#     return impossibleClassesList
#
#
#
# def train_worst_pairs(X_train, y_train):
#     # listOfWorstPairs = [('B', 'F'), ('B', 'T'), ('C', 'O'), ('F', 'T'),
#     #                     ('G', 'O'),('G', 'P'), ('G', 'R'), ('G', 'T'),
#     #                     ('G','U'),('G', 'V'), ('I', 'M'),
#     #                     ('I', 'O'), ('I', 'T'), ('L', 'V'), ('M','S'),
#     #                     ('N', 'P'), ('N', 'Q'),
#     #                     ('R', 'V')]
#
#     listOfWorstPairs = [(2, 6)]
#     # print y_train
#
#     listOfSVMS = []
#     for a,b in listOfWorstPairs:
#
#         if a in y_train and b in y_train:
#             # print a,b
#             # usa dados das duas letras juntas pra treinar svm
#             dataset_a = X_train[y_train == a]
#             dataset_b = X_train[y_train == b]
#
#             rows_a, columns_a = dataset_a.shape
#             rows_b, columns_b = dataset_b.shape
#
#             X = np.vstack((dataset_a, dataset_b))
#
#             y_a = np.ones(rows_a) * a
#             y_b = np.ones(rows_b) * b
#
#             # print y_a.shape
#             # print y_b.shape
#             y = np.concatenate((y_a, y_b), axis=0)
#             # print y
#
#             clf = SVC()
#             clf.fit(X, y)
#
#
#             listOfSVMS.append(clf)
#
#     return listOfSVMS
