import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import graphviz


train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
test_labels = pd.read_csv('./test_result.csv')
test_labels = test_labels['Survived']

print(train_data.info())
print(train_data.describe())
print(train_data.describe(include=['O']))
print(train_data.head())
print(train_data.tail())

print(test_data.info())
print(test_data.describe())
print(test_data.describe(include=['O']))
print(test_data.head())
print(test_data.tail())

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features,train_labels)

test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
pred_labels = clf.predict(test_features)

acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(acc_decision_tree)

print(np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))

print(accuracy_score(test_labels,pred_labels))

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("tree")
graph.view('graph')
