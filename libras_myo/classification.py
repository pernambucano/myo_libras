from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler

def classify(featureMatrix):

    # Feature Projection Using LDA
    # X_train, X_test, y_train, y_test = train_test_split(featureMatrix[:,:63], featureMatrix[:,64], test_size=0.2)
    #
    # minmax = MinMaxScaler(feature_range=(-1,1))
    # lda = LDA(n_components=2).fit(X_train, y_train)
    #
    # X_train_reduced = lda.transform(X_train)
    # X_test_reduced = lda.transform(X_test)
    #
    # print "X_train_reduced is %s" % X_train_reduced
    # print "X_test_reduced is %s" % X_test_reduced

    # Train/Test using KFold cross validation
    splitsNumber = 10
    kf = KFold(n_splits=splitsNumber, shuffle=True)
    mean = 0
    for train_indices, test_indices  in kf.split(featureMatrix[:,:63], featureMatrix[:,64]):

        features_train = [featureMatrix[ii, :63] for ii in train_indices]
        features_test = [featureMatrix[ii, :63] for ii in test_indices]
        classes_train = [featureMatrix[ii, 64] for ii in train_indices]
        classes_test = [featureMatrix[ii, 64] for ii in test_indices]


        # params = {'max_depth':[10,20], 'n_estimators':[10,50,100]}
        # rfr = RandomForestClassifier()
        # classifier = GridSearchCV(rfr, params)
        # classifier.fit(features_train, classes_train)
        # print classifier.best_params_

        classifier = RandomForestClassifier(max_depth=10, n_estimators=100)
        classifier.fit(features_train, classes_train)

        mean += classifier.score(features_test,classes_test)


    return mean/splitsNumber
