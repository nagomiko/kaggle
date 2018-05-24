from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

# print(train_data)

train_data.Sex = train_data.Sex.replace(['male', 'female'], [0, 1])
train_data.Age = train_data.Age.fillna(0.0)
train_data.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Pclass', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    axis=1, inplace=True)

test_data.Sex = test_data.Sex.replace(['male', 'female'], [0, 1])
test_data.Age = test_data.Age.fillna(0.0)
test_data.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Pclass', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    axis=1, inplace=True)

test_data = test_data[['Sex', 'Age']]
test_data.values.astype(np.int)

labels = train_data['Survived']
data = train_data[['Sex', 'Age']]
data.values.astype(np.int)
labels.values.astype(np.int)

print(data)
print(labels)

rates = []
for i in range(10):
    test = np.arange(len(labels)) % 10 == i
    x_te = data[test]
    x_tr = data[~test]
    lb_te = labels[test]
    lb_tr = labels[~test]
    clf = svm.SVC(C=1, kernel='rbf', gamma=1).fit(x_tr, lb_tr)
    rates.append(np.mean(clf.predict(x_te) == lb_te))

print(np.mean(rates))

y_test_pred = clf.predict(test_data)
# test.csvへの予測結果
print(y_test_pred)

print(data.shape)
print(test_data.shape)
print(y_test_pred.shape)

df_out = pd.read_csv("./input/test.csv")
df_out["Survived"] = y_test_pred

# outputディレクトリに出力する
df_out[["PassengerId", "Survived"]].to_csv("./output/submission2.csv", index=False)
