import imblearn as imb
import numpy as np
from sklearn.datasets import make_classification
import collections
import matplotlib.pyplot as plt

# Stworzenie zbioru danych:
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# z
counter = collections.Counter(y)
print(counter)

# wykres przedstawiający nasz niezbalansowany zbiór danych:
for label, _ in counter.items():
	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
