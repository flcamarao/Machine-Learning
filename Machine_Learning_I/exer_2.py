from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tqdm.notebook import trange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('Diamond Price Prediction.csv', nrows=10_000)
df = df.drop(columns=['Cut(Quality)', 'Color', 'Clarity']) # Drop categorical first for speed
display(df)

X = df.drop(columns=['Price(in US dollars)'])
y = df['Price(in US dollars)']

training = pd.DataFrame()
test = pd.DataFrame()
for seedN in trange(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seedN)
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 31, 1)

    for n_neighbors in neighbors_settings:   
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        training_accuracy.append(knn.score(X_train, y_train))
        test_accuracy.append(knn.score(X_test, y_test))
    
    training[seedN] = training_accuracy
    test[seedN] = test_accuracy

fig, ax= plt.subplots(figsize=(8, 5), dpi=300)
plt.errorbar(neighbors_settings, training.mean(axis=1),
             yerr=training.std(axis=1), label="training accuracy")
plt.errorbar(neighbors_settings, test.mean(axis=1),
             yerr=test.std(axis=1), label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.text(0.62, 0.6, 'Optimal n_neighbor', transform=ax.transAxes)
ax.axvspan(15, 30, color="#00FF00", alpha=0.5)
plt.legend()
plt.show()

k_optim = test.mean(axis=1).argmax() + 1
print(f'Optimal n_neighbor: {k_optim}')

df = pd.read_csv('Diamond Price Prediction.csv', nrows=10_000)
df = df.drop(columns=['Cut(Quality)', 'Color', 'Clarity'])
X = df.drop(columns=['Price(in US dollars)'])
y = df['Price(in US dollars)']

training_accuracy = []
test_accuracy = []
for seedN in trange(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seedN)
    knn = KNeighborsRegressor(n_neighbors=k_optim)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

training_score = np.array(training_accuracy).mean()
test_score = np.array(test_accuracy).mean()
print(f'Training Accuracy: {training_score*100:.2f}%')
print(f'Test Accuracy: {test_score*100:.2f}%')

df = pd.read_csv('Diamond Price Prediction.csv')
df = df.drop(columns=['Cut(Quality)', 'Color', 'Clarity'])
X = df.drop(columns=['Price(in US dollars)'])
y = df['Price(in US dollars)']

training_accuracy = []
test_accuracy = []
for seedN in trange(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seedN)
    knn = KNeighborsRegressor(n_neighbors=k_optim)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

training_score = np.array(training_accuracy).mean()
test_score = np.array(test_accuracy).mean()
print(f'Training Accuracy: {training_score*100:.2f}%')
print(f'Test Accuracy: {test_score*100:.2f}%')

df = pd.read_csv('Diamond Price Prediction.csv', nrows=10_000)
display(df)
df = pd.get_dummies(df, columns=['Cut(Quality)', 'Color', 'Clarity'])

X = df.drop(columns=['Price(in US dollars)'])
y = df['Price(in US dollars)']

display(df)

training_accuracy = []
test_accuracy = []
for seedN in trange(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seedN)
    knn = KNeighborsRegressor(n_neighbors=k_optim)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

training_score = np.array(training_accuracy).mean()
test_score = np.array(test_accuracy).mean()
print(f'Training Accuracy: {training_score*100:.2f}%')
print(f'Test Accuracy: {test_score*100:.2f}%')

df = pd.read_csv('Diamond Price Prediction.csv')
df = pd.get_dummies(df, columns=['Cut(Quality)', 'Color', 'Clarity'])
X = df.drop(columns=['Price(in US dollars)'])
y = df['Price(in US dollars)']

training_accuracy = []
test_accuracy = []
for seedN in trange(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seedN)
    knn = KNeighborsRegressor(n_neighbors=k_optim)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

training_score = np.array(training_accuracy).mean()
test_score = np.array(test_accuracy).mean()
print(f'Training Accuracy: {training_score*100:.2f}%')
print(f'Test Accuracy: {test_score*100:.2f}%')

