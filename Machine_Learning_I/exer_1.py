from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

df = pd.read_csv('milknew.csv')
display(df.head())
print(df.shape)

df.describe()

df.info()

df.isna().sum()

label_encoder = LabelEncoder()
label_encoder.fit(df['Grade'])
label_encoder.classes_ = np.array(['low', 'medium', 'high'])
df['Grade'] = label_encoder.transform(df['Grade'])

print(f'The encoded label are: {dict(zip(label_encoder.classes_, np.unique(df["Grade"])))}')
print('\nThe new dataframe with encoded class labels:')
display(df.head())

X = df.drop(columns=['Grade'])
y = df['Grade']

display(X, y)

pcc = 1.25*np.sum(((len(X) / len(y))**2))
print("The 1.25*PCC baseline accuracy for our problem is {:.2f}%.".format(
        pcc*100))

test_size = 0.51
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=1)

print('Check train & test size from train_test_split')
print(f'Train Size: {X_train.shape}')
print(f'Test Size: {X_test.shape}')

print('\nCheck train & test size using manual calculation')
print(f'Train Size: {len(df)*(1-test_size)}')
print(f'Test Size: {len(df)*(test_size)}')

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=1)
print(f'Train Size: {X_train.shape}')
print(f'Test Size: {X_test.shape}')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print(f'Training Score: {knn.score(X_train, y_train)*100:.2f} %')
print(f' Testing Score: {knn.score(X_test, y_test)*100:.2f} %')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print(f'Training Score: {knn.score(X_train_scaled, y_train)*100:.2f} %')
print(f' Testing Score: {knn.score(X_test_scaled, y_test)*100:.2f} %')

columns_name = df.columns
for c in columns_name[:-1]:
    X_1 = df.drop(columns=[c, 'Grade'])
    y_1 = df['Grade']
    test_size = 0.25
    X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=test_size,
                                                        random_state=1)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_1_train, y_1_train)
    print(c)
    print(f'Training Score: {knn.score(X_1_train, y_1_train)*100:.2f} %')
    print(f' Testing Score: {knn.score(X_1_test, y_1_test)*100:.2f} %\n')

training = pd.DataFrame()
test = pd.DataFrame()
for seedN in tqdm(range(1,21)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seedN)
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 21)

    for n_neighbors in neighbors_settings:   
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        training_accuracy.append(knn.score(X_train, y_train))
        test_accuracy.append(knn.score(X_test, y_test))
    
    training[seedN] = training_accuracy
    test[seedN] = test_accuracy

fig, ax= plt.subplots(figsize=(6, 3), dpi=200)
plt.errorbar(neighbors_settings, training.mean(axis=1),
             yerr=training.std(axis=1)/2, label="training accuracy")
plt.errorbar(neighbors_settings, test.mean(axis=1),
             yerr=test.std(axis=1)/10, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.text(0.10, 0.70, 'Best n_neighbor', transform=ax.transAxes)
ax.axvspan(.5, 1.5, color="#00FF00", alpha=0.5)
plt.legend()
plt.show()

print(f'The best average value of test accuracy: {test[1].mean() *100:.2f} % at n_neighbor = 1.')
