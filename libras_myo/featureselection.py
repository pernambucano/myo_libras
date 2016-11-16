import pandas as pd
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

df = pd.io.parsers.read_csv(
    filepath_or_buffer="featureMatrix.csv",
    header=None,
    sep=","
)
#
#
X = df.loc[:, :63].values
y = df.loc[:, 64].values
#
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('./sbs.png', dpi=300)
plt.show()


print sbs.subsets_
k5 = list(sbs.subsets_[-26])
print(df.columns[:][k5])


knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))


# from sklearn.ensemble import RandomForestClassifier
#
# # feat_labels = df.columns[1:]
#
# forest = RandomForestClassifier(n_estimators=10000,
#                                 random_state=0,
#                                 n_jobs=-1)
#
# forest.fit(X_train, y_train)
# importances = forest.feature_importances_
#
# indices = np.argsort(importances)[::-1]
#
# for f in range(X_train.shape[1]):
#     print("%2d) %f" % (f + 1, importances[indices[f]]))
#
# plt.title('Feature Importances')
# plt.bar(range(X_train.shape[1]),
#         importances[indices],
#         color='lightblue',
#         align='center')
#
# plt.xticks(range(X_train.shape[1]),
#            indices, rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# #plt.savefig('./random_forest.png', dpi=300)
# plt.show()
