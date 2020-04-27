import sys
import scipy
import pandas as pd
import numpy as np
import matplotlib as mt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


###############
# Summarize
###############

dt = pd.read_csv("../data/processed/iris_processed.csv")
print(dt)
print(dt.shape)
print(dt.describe())

print(dt.groupby("variety").size())


###############
# Visualize
###############

dt.plot(kind = "box", subplots = True, layout = (2,2), sharex = False, sharey = False)
mt.pyplot.show()

dt.hist()
mt.pyplot.show()

pd.plotting.scatter_matrix(dt)
mt.pyplot.show()

###############
# Evaluate algorithms
###############

###
# split
###
array = dt.values
X = array[:, 0:4]
y = array[:,4]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2)

###
# model
###

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
mt.pyplot.boxplot(results, labels=names)
mt.pyplot.title('Algorithm Comparison')
mt.pyplot.show()


###############
# Predictions
###############

model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

###
# Evaluate predictions
###
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
