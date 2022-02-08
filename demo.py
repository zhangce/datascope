
### TO RUN THIS MINIMAL DEMO ###
# pip3 install -r requirements.txt
# python3 setup.py install
# python3 demo.py

import sys  
from pathlib import Path  
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

import numpy as np
from copy import deepcopy

from datascope.importance.common import SklearnModelUtility, binarize, get_indices
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod

from experiments.dataset import Dataset
from experiments.pipelines import Pipeline, get_model, ModelType

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

# Load and split data
#
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Inject noise
#
X_train_dirty = deepcopy(X_train)
y_train_dirty = deepcopy(y_train)
y_train_dirty = 2 - y_train_dirty

# Setup target model, utility function
#
model = get_model(ModelType.LogisticRegression)
utility = SklearnModelUtility(model, accuracy_score)

# Compute Importance
#
method = ImportanceMethod.NEIGHBOR
importance = ShapleyImportance(method=method, utility=utility)

importances = importance.fit(X_train_dirty, y_train_dirty).score(X_test, y_test)

# Order data examples by their importances
#
ordered_examples = np.argsort(importances)

print("DataScope recommend to fix examples in the following order:", ordered_examples)

import numpy.random as random
ordered_examples_random = np.argsort(random.random(importances.shape[0]))

print("Random recommend to fix examples in the following order:", ordered_examples_random)

y_train_random = deepcopy(y_train_dirty)
y_train_shapley = deepcopy(y_train_dirty)

# Fix one by one
#
for i in range(ordered_examples.shape[0]):

	# current Shapley model 
	clf = LogisticRegression(random_state=0).fit(X_train_dirty, y_train_shapley)
	shapley_score = clf.score(X_test, y_test)

	clf = LogisticRegression(random_state=0).fit(X_train_dirty, y_train_random)
	random_score = clf.score(X_test, y_test)

	print("")
	print("current model accuracy: DataScope score", shapley_score, "     Random score", random_score)

	# fix a label, following shapley
	print(1.0*i/ordered_examples.shape[0], "DataScope: fix example", i, "by flipping label from", y_train_shapley[i], "to", y_train[i])
	y_train_shapley[ordered_examples[i]] = y_train[ordered_examples[i]]

	# fix a label, following shapley
	print(1.0*i/ordered_examples.shape[0], "Random: fix example", i, "by flipping label from", y_train_random[i], "to", y_train[i])
	y_train_random[ordered_examples_random[i]] = y_train[ordered_examples_random[i]]










