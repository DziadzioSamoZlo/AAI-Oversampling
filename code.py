#%%
import imblearn as imb
import numpy as np
from sklearn.datasets import make_classification
import collections
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Stworzenie zbioru danych:
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)

def plotMaker(X, y):
    ### Wykres przedstawiający zbiory
    counter = collections.Counter(y)
    print(counter)
    for label, _ in counter.items():
        row_ix = np.where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()
    classify(X, y)

def classify(X, y):
    ### Funkcja obliczajaca punktacje poszczególnych zbiorów
    model = DecisionTreeClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2137)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % np.mean(scores))
    print()

# Stworzenie modeli testowych:
from imblearn.over_sampling import RandomOverSampler

over_sampler = RandomOverSampler()
x_train_over, y_train_over = over_sampler.fit_resample(X, y)

from imblearn.over_sampling import SMOTE

smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(X, y)

from imblearn.over_sampling import ADASYN

adasyn = ADASYN()
x_train_adasyn, y_train_adasyn = adasyn.fit_resample(X, y)

from imblearn.over_sampling import SMOTENC

smote = SMOTENC([1])
x_train_smotenc, y_train_smotenc = smote.fit_resample(X, y)

from imblearn.over_sampling import BorderlineSMOTE

smote = BorderlineSMOTE(kind = 'borderline-1')
x_train_bsmote, y_train_bsmote = smote.fit_resample(X, y)

from imblearn.over_sampling import SVMSMOTE

smote = SVMSMOTE()
x_train_smotesvm, y_train_smotesvm = smote.fit_resample(X, y)

# Przedstawienie danych
print("Base line")
plotMaker(X, y)
print("Random Oversampling")
plotMaker(x_train_over, y_train_over)
print("SMOTE")
plotMaker(x_train_smote, y_train_smote)
print("ADASYN")
plotMaker(x_train_adasyn, y_train_adasyn)
print("SMOTENC")
plotMaker(x_train_smotenc, y_train_smotenc)
print("BorderlineSMOTE")
plotMaker(x_train_bsmote, y_train_bsmote)
print("SMOTE SVM")
plotMaker(x_train_smotesvm, y_train_smotesvm)
