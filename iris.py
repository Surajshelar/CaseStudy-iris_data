import pandas as pd

data = pd.read_csv("iris.data")
print(data)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

lgr = LogisticRegression(random_state=0)
rf = RandomForestClassifier(random_state=1)
dt = DecisionTreeClassifier(random_state=0)
sc = svm.SVC()
nb = MultinomialNB()
grb = GradientBoostingClassifier(n_estimators=10)

x = data.drop('Species', axis=1)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2 )

lgr.fit(x_train, y_train)
rf.fit(x_train, y_train)
dt.fit(x_train, y_train)
sc.fit(x_train, y_train)
grb.fit(x_train, y_train)
nb.fit(x_train, y_train)

logr_predict = lgr.predict(x_test)
rfc_predict = rf.predict(x_test)
dtc_predict = dt.predict(x_test)
svm_predict = sc.predict(x_test)
grbc_predict = grb.predict(x_test)
nbc_predict = nb.predict(x_test)


print('Random_Forest:', accuracy_score(y_test, rfc_predict))
print('Decision_Tree:', accuracy_score(y_test, dtc_predict))
print('LogisticRegression:', accuracy_score(y_test, logr_predict))
print('NaiveBayes:', accuracy_score(y_test,  nbc_predict))
print('SupportVector:', accuracy_score(y_test, svm_predict))
print('Gradient Boosting:', accuracy_score(y_test,  grbc_predict))




'''
Random_Forest: 0.9666666666666667
Decision_Tree: 0.9666666666666667
LogisticRegression: 0.9666666666666667
NaiveBayes: 0.5666666666666667
SupportVector: 0.9666666666666667
Gradient Boosting: 0.9666666666666667
'''