import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pylab as plot
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

import warnings
warnings.filterwarnings("ignore")  

import warnings
def showwarning(*args, **kwargs):
    if args[1] is DeprecationWarning:
        return
    warnings._showwarning_orig(*args, **kwargs)
warnings.showwarning = showwarning

boston = load_boston()
df_boston = pd.DataFrame.from_dict(boston['data'])  # Feature variables
df_boston['Target'] = boston['target']
df_features = df_boston.drop(columns=['Target'])
df_boston.head()

No_Trials = 20  # Number of Trials
lr_training_accuracy = []
lr_test_accuracy = []
for seedN in range(No_Trials):
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_boston['Target'], test_size=0.25, random_state=seedN)
    lr = LinearRegression()
    lr.fit(X_train, y_train)  # build the model
    # record training set accuracy
    lr_training_accuracy.append(lr.score(X_train, y_train))
    # record generalization accuracy
    lr_test_accuracy.append(lr.score(X_test, y_test))

print("TRAIN SET: Mean = ", np.mean(lr_training_accuracy),
      " ; Stdev = ", np.std(lr_training_accuracy))
print(" TEST SET: Mean = ", np.mean(lr_test_accuracy),
      " ; Stdev = ", np.std(lr_test_accuracy))

No_Trials = 20  # Number of Trials
alpha_ridge = [0.000001, 0.001, 0.1, 1, 10, 100]
ridge_all_training = pd.DataFrame(alpha_ridge, columns=['alpha'])
ridge_all_test = pd.DataFrame(alpha_ridge, columns=['alpha'])

for seedN in range(No_Trials):
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_boston['Target'], test_size=0.25, random_state=seedN)
    training_accuracy = []
    test_accuracy = []

    for alpha_run in alpha_ridge:
        reg = Ridge(alpha=alpha_run)
        reg.fit(X_train, y_train)  # build the model
        # record training set accuracy
        training_accuracy.append(reg.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(reg.score(X_test, y_test))
    ridge_all_training[seedN] = training_accuracy
    ridge_all_test[seedN] = test_accuracy

ridge_all_test['mean'] = ridge_all_test.drop(columns=['alpha']).mean(axis=1)
ridge_all_test['std'] = ridge_all_test.drop(
    columns=['alpha', 'mean']).std(axis=1)

ridge_all_training['mean'] = ridge_all_training.drop(
    columns=['alpha']).mean(axis=1)
ridge_all_training['std'] = ridge_all_training.drop(
    columns=['alpha', 'mean']).std(axis=1)
print(
    f"The optimal alpha = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'std']}")

fig = plt.figure(figsize=(20, 6))
plt.xscale('log')
params = {'legend.fontsize': 20, 'legend.handlelength': 2}
plot.rcParams.update(params)
plt.plot(alpha_ridge, ridge_all_training['mean'].to_list(),
         label="training accuracy",
         color='blue', marker='o',
         linestyle='dashed', markersize=18)
plt.plot(alpha_ridge, ridge_all_test['mean'].to_list(),
         label="test accuracy",
         color='red', marker='^',
         linestyle='-', markersize=18)
plt.title("Training and Test Accuracy vs Alpha", fontsize=18)
plt.ylabel("Accuracy, $R^2$", fontsize=18)
plt.xlabel("alpha_ridge", fontsize=18)
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 6))
best_alpha = ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'alpha']
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_boston['Target'], test_size=0.25, random_state=1)
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
plt.plot(lr.coef_, '-^', label=f"LR")
plt.plot(ridge.coef_, '--x', label=f"Ridge alpha({best_alpha})")
plt.plot(ridge10.coef_, '--*', label=f"Ridge alpha(10)")
plt.title("Regression Coefficient", fontsize=18)
plt.xlabel("Feature", fontsize=18)
plt.ylabel("Coefficient magnitude", fontsize=18)
plt.legend()
plt.show()

weights_normalized = ridge.coef_/np.sum(np.abs(ridge.coef_))
fig = plt.figure(figsize=(10, 5))
n_features = 13
plt.barh(range(n_features), weights_normalized,
         align='center', color='indianred')
plt.yticks(np.arange(n_features), df_features.columns)
# plt.yticks(np.arange(n_features))
plt.title("Top Predictor", fontsize=18)
plt.xlabel("Feature Importance", fontsize=18)
plt.ylabel("Features", fontsize=18)
plt.ylim(-1, n_features)
plt.show()
print("Weight of the top predictor = %f" %np.amax(np.abs(ridge.coef_)))
print("Top Predictor is Column %s" %np.abs(np.argmax(ridge.coef_)))

No_Trials = 20  # Number of Trials
alpha_lasso = [0.000001, 0.001, 0.1, 1, 10, 100]
lasso_all_training = pd.DataFrame(alpha_lasso, columns=['alpha'])
lasso_all_test = pd.DataFrame(alpha_lasso, columns=['alpha'])

for seedN in range(No_Trials):
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_boston['Target'], test_size=0.25, random_state=seedN)
    training_accuracy = []
    test_accuracy = []

    for alpha_run in alpha_lasso:
        lasso = Lasso(alpha=alpha_run, max_iter=1_000_000)
        lasso.fit(X_train, y_train)  # build the model
        # record training set accuracy
        training_accuracy.append(lasso.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(lasso.score(X_test, y_test))
    lasso_all_training[seedN] = training_accuracy
    lasso_all_test[seedN] = test_accuracy

lasso_all_test['mean'] = lasso_all_test.drop(columns=['alpha']).mean(axis=1)
lasso_all_test['std'] = lasso_all_test.drop(
    columns=['alpha', 'mean']).std(axis=1)

lasso_all_training['mean'] = lasso_all_training.drop(
    columns=['alpha']).mean(axis=1)
lasso_all_training['std'] = lasso_all_training.drop(
    columns=['alpha', 'mean']).std(axis=1)
print(
    f"The optimal alpha = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'std']}")

fig = plt.figure(figsize=(20, 6))
plt.xscale('log')
params = {'legend.fontsize': 20, 'legend.handlelength': 2}
plot.rcParams.update(params)
plt.plot(alpha_lasso, lasso_all_training['mean'].to_list(),
         label="training accuracy",
         color='blue', marker='o',
         linestyle='dashed', markersize=18)
plt.plot(alpha_lasso, lasso_all_test['mean'].to_list(),
         label="test accuracy",
         color='red', marker='^',
         linestyle='-', markersize=18)
plt.title("Training and Test Accuracy vs Alpha", fontsize=18)
plt.ylabel("Accuracy, $R^2$", fontsize=18)
plt.xlabel("alpha_lasso", fontsize=18)
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 6))
best_alpha = lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'alpha']
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_boston['Target'], test_size=0.25, random_state=1)
lr = LinearRegression().fit(X_train, y_train)
lasso = Lasso(alpha=best_alpha).fit(X_train, y_train)
lasso10 = Lasso(alpha=10).fit(X_train, y_train)
plt.plot(lr.coef_, '-^', label=f"LR")
plt.plot(lasso.coef_, '--x', label=f"Lasso alpha({best_alpha})")
plt.plot(lasso10.coef_, '--*', label=f"Lasso alpha(10)")
plt.title("Regression Coefficient", fontsize=18)
plt.xlabel("Feature", fontsize=18)
plt.ylabel("Coefficient magnitude", fontsize=18)
plt.legend()
plt.show()

weights_normalized = lasso.coef_/np.sum(np.abs(lasso.coef_))
fig = plt.figure(figsize=(10, 5))
n_features = 13
plt.barh(range(n_features), weights_normalized,
         align='center', color='indianred')
plt.yticks(np.arange(n_features), df_features.columns)
plt.title("Top Predictor", fontsize=18)
plt.xlabel("Feature Importance", fontsize=18)
plt.ylabel("Features", fontsize=18)
plt.ylim(-1, n_features)
plt.show()
print("Weight of the top predictor = %f" %np.amax(np.abs(lasso.coef_)))
print("Top Predictor is Column %s" %np.abs(np.argmax(lasso.coef_)))

print('Linear Regression')

print("TRAIN SET: Mean = ", np.mean(lr_training_accuracy),
      " ; Stdev = ", np.std(lr_training_accuracy))
print(" TEST SET: Mean = ", np.mean(lr_test_accuracy),
      " ; Stdev = ", np.std(lr_test_accuracy))
print('\nRidge Regularization')
print(
    f"The optimal alpha = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'std']}")
print('\nLasso Regularization')
print(
    f"The optimal alpha = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'std']}")

df_parkinsons = pd.read_excel(
    './parkinsons_updrs.xlsx', sheet_name="parkinsons_updrs")
df_parkinsons_features = df_parkinsons.columns[:-1]
df_features = df_parkinsons.drop(columns=['Target'])  # value for X
df_parkinsons.head()

No_Trials = 30  # Number of Trials
lr_training_accuracy = []
lr_test_accuracy = []
for seedN in range(No_Trials):
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_parkinsons['Target'], test_size=0.25,
        random_state=seedN)
    lr = LinearRegression()
    lr.fit(X_train, y_train)  # build the model
    # record training set accuracy
    lr_training_accuracy.append(lr.score(X_train, y_train))
    # record generalization accuracy
    lr_test_accuracy.append(lr.score(X_test, y_test))

print("TRAIN SET: Mean = ", np.mean(lr_training_accuracy),
      " ; Stdev = ", np.std(lr_training_accuracy))
print(" TEST SET: Mean = ", np.mean(lr_test_accuracy),
      " ; Stdev = ", np.std(lr_test_accuracy))

No_Trials = 30  # Number of Trials
alpha_ridge = [.000001, .001, .1, 1, 10, 100]
ridge_all_training = pd.DataFrame(alpha_ridge, columns=['alpha'])
ridge_all_test = pd.DataFrame(alpha_ridge, columns=['alpha'])

for seedN in range(No_Trials):
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_parkinsons['Target'], test_size=0.25,
        random_state=seedN)
    training_accuracy = []
    test_accuracy = []

    for alpha_run in alpha_ridge:
        reg = Ridge(alpha=alpha_run)
        reg.fit(X_train, y_train)  # build the model
        # record training set accuracy
        training_accuracy.append(reg.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(reg.score(X_test, y_test))
    ridge_all_training[seedN] = training_accuracy
    ridge_all_test[seedN] = test_accuracy

ridge_all_test['mean'] = ridge_all_test.drop(columns=['alpha']).mean(axis=1)
ridge_all_test['std'] = ridge_all_test.drop(
    columns=['alpha', 'mean']).std(axis=1)

ridge_all_training['mean'] = ridge_all_training.drop(
    columns=['alpha']).mean(axis=1)
ridge_all_training['std'] = ridge_all_training.drop(
    columns=['alpha', 'mean']).std(axis=1)
print(
    f"The optimal alpha = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'std']}")

fig = plt.figure(figsize=(20, 6))
plt.xscale('log')
params = {'legend.fontsize': 20, 'legend.handlelength': 2}
plot.rcParams.update(params)
plt.plot(alpha_ridge, ridge_all_training['mean'].to_list(),
         label="training accuracy",
         color='blue', marker='o',
         linestyle='dashed', markersize=18)
plt.plot(alpha_ridge, ridge_all_test['mean'].to_list(),
         label="test accuracy",
         color='red', marker='^',
         linestyle='-', markersize=18)
plt.title("Training and Test Accuracy vs Alpha", fontsize=18)
plt.ylabel("Accuracy, $R^2$", fontsize=18)
plt.xlabel("alpha_ridge", fontsize=18)
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 6))
best_alpha = ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'alpha']
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_parkinsons['Target'], test_size=0.25, random_state=1)
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
plt.plot(lr.coef_, '-^', label=f"LR")
plt.plot(ridge.coef_, '--x', label=f"Ridge alpha({best_alpha})")
plt.plot(ridge10.coef_, '--*', label=f"Ridge alpha(10)")
plt.title("Regression Coefficient", fontsize=18)
plt.xlabel("Feature", fontsize=18)
plt.ylabel("Coefficient magnitude", fontsize=18)
plt.legend()
plt.show()

weights_normalized = ridge.coef_/np.sum(np.abs(ridge.coef_))
fig = plt.figure(figsize=(10, 5))
n_features = 21
plt.barh(range(n_features), weights_normalized,
         align='center', color='indianred')
plt.yticks(np.arange(n_features), df_parkinsons_features)
plt.title("Top Predictor", fontsize=18)
plt.xlabel("Feature Importance", fontsize=18)
plt.ylabel("Features", fontsize=18)
plt.ylim(-1, n_features)
plt.show()
print("Weight of the top predictor = %f" %np.amax(np.abs(ridge.coef_)))
print("Top Predictor is Column %s" %np.abs(np.argmax(ridge.coef_)))

No_Trials = 2  # Number of Trials Limited for a reasonable runtime
alpha_lasso = [.000001, .001, .1, 1, 10, 100]
lasso_all_training = pd.DataFrame(alpha_lasso, columns=['alpha'])
lasso_all_test = pd.DataFrame(alpha_lasso, columns=['alpha'])

for seedN in trange(No_Trials):
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_parkinsons['Target'], test_size=0.25,
        random_state=seedN)
    training_accuracy = []
    test_accuracy = []

    for alpha_run in tqdm(alpha_lasso):
        lasso = Lasso(alpha=alpha_run, max_iter=100_000)
        lasso.fit(X_train, y_train)  # build the model
        # record training set accuracy
        training_accuracy.append(lasso.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(lasso.score(X_test, y_test))
    lasso_all_training[seedN] = training_accuracy
    lasso_all_test[seedN] = test_accuracy

lasso_all_test['mean'] = lasso_all_test.drop(columns=['alpha']).mean(axis=1)
lasso_all_test['std'] = lasso_all_test.drop(
    columns=['alpha', 'mean']).std(axis=1)

lasso_all_training['mean'] = lasso_all_training.drop(
    columns=['alpha']).mean(axis=1)
lasso_all_training['std'] = lasso_all_training.drop(
    columns=['alpha', 'mean']).std(axis=1)
print(
    f"The optimal alpha = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'std']}")

fig = plt.figure(figsize=(20, 6))
plt.xscale('log')
params = {'legend.fontsize': 20, 'legend.handlelength': 2}
plot.rcParams.update(params)
plt.plot(alpha_lasso, lasso_all_training['mean'].to_list(),
         label="training accuracy",
         color='blue', marker='o',
         linestyle='dashed', markersize=18)
plt.plot(alpha_lasso, lasso_all_test['mean'].to_list(),
         label="test accuracy",
         color='red', marker='^',
         linestyle='-', markersize=18)
plt.title("Training and Test Accuracy vs Alpha", fontsize=18)
plt.ylabel("Accuracy, $R^2$", fontsize=18)
plt.xlabel("alpha_lasso", fontsize=18)
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 6))
best_alpha = lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'alpha']
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_parkinsons['Target'], test_size=0.25, random_state=1)
lr = LinearRegression().fit(X_train, y_train)
lasso = Lasso(alpha=best_alpha).fit(X_train, y_train)
lasso10 = Lasso(alpha=10).fit(X_train, y_train)
plt.plot(lr.coef_, '-^', label=f"LR")
plt.plot(lasso.coef_, '--x', label=f"Lasso alpha({best_alpha})")
plt.plot(lasso10.coef_, '--*', label=f"Lasso alpha(10)")
plt.title("Regression Coefficient", fontsize=18)
plt.xlabel("Feature", fontsize=18)
plt.ylabel("Coefficient magnitude", fontsize=18)
plt.legend()
plt.show()

weights_normalized = lasso.coef_/np.sum(np.abs(lasso.coef_))
fig = plt.figure(figsize=(10, 5))
n_features = 21
plt.barh(range(n_features), weights_normalized,
         align='center', color='indianred')
plt.yticks(np.arange(n_features), df_parkinsons_features)
plt.title("Top Predictor", fontsize=18)
plt.xlabel("Feature Importance", fontsize=18)
plt.ylabel("Features", fontsize=18)
plt.ylim(-1, n_features)
plt.show()
print("Weight of the top predictor = %f" %np.amax(np.abs(lasso.coef_)))
print("Top Predictor is Column %s" %np.abs(np.argmax(lasso.coef_)))

print('Linear Regression')
print("TRAIN SET: Mean = ", np.mean(lr_training_accuracy),
      " ; Stdev = ", np.std(lr_training_accuracy))
print(" TEST SET: Mean = ", np.mean(lr_test_accuracy),
      " ; Stdev = ", np.std(lr_test_accuracy))
print('\nRidge Regularization')
print(
    f"The optimal alpha = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_training.loc[ridge_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{ridge_all_test.loc[ridge_all_test['mean'].argmax(), 'std']}")
print('\nLasso Regularization')
print(
    f"The optimal alpha = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'alpha']}")
print(
    f"TRAIN SET: Mean = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_training.loc[lasso_all_test['mean'].argmax(), 'std']}")
print(
    f" TEST SET: Mean = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'mean']}"
    f" ; Stdev = "
    f"{lasso_all_test.loc[lasso_all_test['mean'].argmax(), 'std']}")
