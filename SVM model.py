import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Cactei = 0, Pancake plant = 1, Sanseveria = 2, Succulent = 3

labelEncoder = LabelEncoder()
data = pd.read_csv("C:/Users/anna/PycharmProjects/AquaFlora-Forecast/PlantDataLabels.csv", sep=';')
# print(data.dtypes)
data['Waterneed'] = pd.to_numeric(data['Waterneed'])
data['Plant Type'] = labelEncoder.fit_transform(data['Plant Type'])
print(labelEncoder.classes_)
transformed = labelEncoder.transform(['Cactei', 'Pancake Plant', 'Sanseveria', 'Succulent'])
print(transformed)
# print(data.dtypes)

X = data.drop(columns=['Unnamed: 0', 'Waterneed', 'Time'])
y = data['Waterneed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_data = [[508.0, 23.8, 55.0, 17.0, 3]]

svr = SVR(kernel='rbf', C=10, gamma='auto')
svr.fit(X_train, y_train)
acc = svr.score(X_test, y_test)
print(acc)
prediction = svr.predict(test_data)
# print(X_test.shape)
# print(X_test)
print(prediction)
'''

'''


''''
parameters = {'kernel': ('rbf', 'sigmoid'), 'C': [0.1, 1, 10], 'gamma': ('auto', 'scale')}
svr = SVR()
clf = GridSearchCV(svr, parameters, scoring='neg_mean_squared_error', verbose=14)
clf.fit(X_train, y_train)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_score_)
'''


