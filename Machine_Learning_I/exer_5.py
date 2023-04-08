import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from tqdm.notebook import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb 
from sklearn.ensemble import AdaBoostRegressor 
import lightgbm as ltb 
ad_csv('Bike_Sharing_hour.csv')
display(rides.head())
display(rides.columns)
print(rides.shape)

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
display(data.head(100))
print(data.columns)

quant_features = ['cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
display(data)

test_data = data[-21*24:]  # Save the last 21 days as test set
data = data[:-21*24]    # All other data except the last 21 days as training set

target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields] # Training Set
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields] # Test Set

val_set =60*24 # in days*hr
train_features, train_targets = features[:-val_set], targets[:-val_set]
val_features, val_targets = features[-val_set:], targets[-val_set:]

X_train=train_features
y_train=train_targets['cnt']
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

X_validation=val_features
y_validation=val_targets['cnt']
print(f'\nX_validation shape: {X_validation.shape}')
print(f'y_validation shape: {y_validation.shape}')

X_test=test_features
y_test=test_targets['cnt']
print(f'\nX_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

def lr(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # NO Hyperparamter tuning

    # Start time
    start_time = time.time()

    # Training
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    lr_train = lr.score(X_train, y_train)
    lr_validation = lr.score(X_validation, y_validation)
    lr_test = lr.score(X_test, y_test)

    # Top Predictor
    lr_coef = lr.coef_
    top_predictor = X_train.columns[np.argmax(np.abs(lr_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(lr_coef))
    ax.barh(X_train.columns[sorted_index], abs(lr_coef)[sorted_index])
    ax.set_title('Linear Regression Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['Linear Regression', lr_train, lr_validation, lr_test, 'NA',
               top_predictor, run_time]

    return results

def l1(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_alpha = np.array([1e-5, 1e-4, 0.001, 0.01, 0.05,
                          0.1, 0.5, 1, 10, 100])
    validation_scores = []
    for alpha in tqdm(arr_alpha):
        l1 = Lasso(max_iter=100_000, alpha=alpha)
        l1.fit(X_train, y_train)
        validation_scores.append(l1.score(X_validation, y_validation))
    best_params = {'alpha': arr_alpha[np.argmax(validation_scores)]}

    # Start time
    start_time = time.time()

    # Training
    l1 = Lasso(max_iter=100_000, alpha=best_params['alpha'])
    l1.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    l1_train = l1.score(X_train, y_train)
    l1_validation = l1.score(X_validation, y_validation)
    l1_test = l1.score(X_test, y_test)

    # Top Predictor
    l1_coef = l1.coef_
    top_predictor = X_train.columns[np.argmax(np.abs(l1_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(l1_coef))
    ax.barh(X_train.columns[sorted_index], abs(l1_coef)[sorted_index])
    ax.set_title('LR w/ Lasso Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['LR w/ Lasso', l1_train, l1_validation, l1_test,
               f"alpha={best_params['alpha']}", top_predictor, run_time]

    return results

def l2(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_alpha = np.array([1e-5, 1e-4, 0.001, 0.01, 0.05,
                          0.1, 0.5, 1, 10, 100])
    validation_scores = []
    for alpha in tqdm(arr_alpha):
        l2 = Ridge(alpha=alpha)
        l2.fit(X_train, y_train)
        validation_scores.append(l2.score(X_validation, y_validation))
    best_params = {'alpha': arr_alpha[np.argmax(validation_scores)]}

    # Start time
    start_time = time.time()

    # Training
    l2 = Ridge(alpha=best_params['alpha'])
    l2.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    l2_train = l2.score(X_train, y_train)
    l2_validation = l2.score(X_validation, y_validation)
    l2_test = l2.score(X_test, y_test)

    # Top Predictor
    l2_coef = l2.coef_
    top_predictor = X_train.columns[np.argmax(np.abs(l2_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(l2_coef))
    ax.barh(X_train.columns[sorted_index], abs(l2_coef)[sorted_index])
    ax.set_title('LR w/ Ridge Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['LR w/ Ridge', l2_train, l2_validation, l2_test,
               f"alpha={best_params['alpha']}", top_predictor, run_time]

    return results

def knn(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_n_neighbors = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    validation_scores = []
    for n_neighbors in tqdm(arr_n_neighbors):
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        validation_scores.append(knn.score(X_validation, y_validation))
    best_params = {
        'n_neighbors': arr_n_neighbors[np.argmax(validation_scores)]}

    # Start time
    start_time = time.time()

    # Training
    knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'])
    knn.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    knn_train = knn.score(X_train, y_train)
    knn_validation = knn.score(X_validation, y_validation)
    knn_test = knn.score(X_test, y_test)

    # NO Top Predictor

    # NO Feature importance plot

    # Summary results
    results = ['KNN', knn_train, knn_validation, knn_test,
               f"n_neighbors={best_params['n_neighbors']}", 'NA', run_time]

    return results

def dt(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_max_depth = np.arange(1,50,2)
    validation_scores = []
    for max_depth in tqdm(arr_max_depth):
        dt = DecisionTreeRegressor(max_depth=max_depth)
        dt.fit(X_train, y_train)
        validation_scores.append(dt.score(X_validation, y_validation))
    best_params = {
        'max_depth': arr_max_depth[np.argmax(validation_scores)]}

    # Start time
    start_time = time.time()

    # Training
    dt = DecisionTreeRegressor(max_depth=best_params['max_depth'])
    dt.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    dt_train = dt.score(X_train, y_train)
    dt_validation = dt.score(X_validation, y_validation)
    dt_test = dt.score(X_test, y_test)

    # Top Predictor
    dt_coef = dt.feature_importances_
    top_predictor = X_train.columns[np.argmax(np.abs(dt_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(dt_coef))
    ax.barh(X_train.columns[sorted_index], abs(dt_coef)[sorted_index])
    ax.set_title('Decision Tree Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['Decision Tree', dt_train, dt_validation, dt_test,
               f"max_depth={best_params['max_depth']}",
               top_predictor, run_time]

    return results

def rf(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_max_depth = np.array([10, 30, 50, 70, 90])
    arr_max_features = np.array([0.3, 0.5, 0.7, 0.9])
    p = itertools.product(arr_max_depth, arr_max_features)
    params = [i for i in p]

    validation_scores = []
    for param in tqdm(params):
        rf = RandomForestRegressor(max_depth=param[0], max_features=param[1],
                                   n_jobs=-1)
        rf.fit(X_train, y_train)
        validation_scores.append(rf.score(X_validation, y_validation))
    optim_params = params[np.argmax(validation_scores)]
    best_params = {
        'max_depth': optim_params[0],
        'max_features': optim_params[1]}

    # Start time
    start_time = time.time()

    # Training
    rf = RandomForestRegressor(max_depth=best_params['max_depth'],
                               max_features=best_params['max_features'])
    rf.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    rf_train = rf.score(X_train, y_train)
    rf_validation = rf.score(X_validation, y_validation)
    rf_test = rf.score(X_test, y_test)

    # Top Predictor
    rf_coef = rf.feature_importances_
    top_predictor = X_train.columns[np.argmax(np.abs(rf_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(rf_coef))
    ax.barh(X_train.columns[sorted_index], abs(rf_coef)[sorted_index])
    ax.set_title('Random Forest Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['Random Forest', rf_train, rf_validation, rf_test,
               (f"max_depth={best_params['max_depth']}, "
                f"max_features={best_params['max_features']}"),
               top_predictor, run_time]

    return results

def gbm(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_max_depth = np.array([5, 7, 10, 20])
    arr_max_features = np.array([0.5, 0.7, 0.9])
    arr_learning_rate = np.array([0.01, 0.1, 1])
    p = itertools.product(arr_max_depth, arr_max_features, arr_learning_rate)
    params = [i for i in p]

    validation_scores = []
    for param in tqdm(params):
        gbm = GradientBoostingRegressor(n_estimators=200,
                                        max_depth=param[0],
                                        max_features=param[1],
                                        learning_rate=param[2])
        gbm.fit(X_train, y_train)
        validation_scores.append(gbm.score(X_validation, y_validation))
    optim_params = params[np.argmax(validation_scores)]
    best_params = {
        'max_depth': optim_params[0],
        'max_features': optim_params[1],
        'learning_rate': optim_params[2]}

    # Start time
    start_time = time.time()

    # Training
    gbm = GradientBoostingRegressor(n_estimators=200,
                                    max_depth=best_params['max_depth'],
                                    max_features=best_params['max_features'],
                                    learning_rate=best_params['learning_rate'])
    gbm.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    gbm_train = gbm.score(X_train, y_train)
    gbm_validation = gbm.score(X_validation, y_validation)
    gbm_test = gbm.score(X_test, y_test)

    # Top Predictor
    gbm_coef = gbm.feature_importances_
    top_predictor = X_train.columns[np.argmax(np.abs(gbm_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(gbm_coef))
    ax.barh(X_train.columns[sorted_index], abs(gbm_coef)[sorted_index])
    ax.set_title('GBM Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['GBM', gbm_train, gbm_validation, gbm_test,
               (f"max_depth={best_params['max_depth']}, "
                f"max_features={best_params['max_features']}, "
                f"learning_rate={best_params['learning_rate']}"),
               top_predictor, run_time]

    return results

def xg(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_max_depth = np.array([5, 10])
    arr_subsample = np.array([0.7, 0.9])
    arr_eta = np.array([0.01, 0.1])
    p = itertools.product(arr_max_depth, arr_subsample, arr_eta)
    params = [i for i in p]

    validation_scores = []
    for param in tqdm(params):
        xg = xgb.XGBRegressor(n_estimators=100, max_depth=param[0],
                              subsample=param[1], eta=param[2])
        xg.fit(X_train, y_train)
        validation_scores.append(xg.score(X_validation, y_validation))
    optim_params = params[np.argmax(validation_scores)]
    best_params = {
        'max_depth': optim_params[0],
        'subsample': optim_params[1],
        'eta': optim_params[2]}

    # Start time
    start_time = time.time()

    # Training
    xg = xgb.XGBRegressor(n_estimators=100,
                          max_depth=best_params['max_depth'],
                          subsample=best_params['subsample'],
                          eta=best_params['eta'])
    xg.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    xg_train = xg.score(X_train, y_train)
    xg_validation = xg.score(X_validation, y_validation)
    xg_test = xg.score(X_test, y_test)

    # Top Predictor
    xg_coef = xg.feature_importances_
    top_predictor = X_train.columns[np.argmax(np.abs(xg_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(xg_coef))
    ax.barh(X_train.columns[sorted_index], abs(xg_coef)[sorted_index])
    ax.set_title('XGBoost Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['XGBoost', xg_train, xg_validation, xg_test,
               (f"max_depth={best_params['max_depth']}, "
                f"subsample={best_params['subsample']}, "
                f"eta={best_params['eta']}"),
               top_predictor, run_time]

    return results

def ada(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_n_estimators = np.array([100, 200, 300])
    arr_learning_rate = np.array([0.01, 0.1, 1])
    p = itertools.product(arr_n_estimators, arr_learning_rate)
    params = [i for i in p]

    validation_scores = []
    for param in tqdm(params):
        ada = AdaBoostRegressor(n_estimators=param[0],
                                learning_rate=param[1])
        ada.fit(X_train, y_train)
        validation_scores.append(ada.score(X_validation, y_validation))
    optim_params = params[np.argmax(validation_scores)]
    best_params = {'n_estimators': optim_params[0],
                   'learning_rate': optim_params[1]}

    # Start time
    start_time = time.time()

    # Training
    ada = AdaBoostRegressor(n_estimators=best_params['n_estimators'],
                            learning_rate=best_params['learning_rate'])
    ada.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    ada_train = ada.score(X_train, y_train)
    ada_validation = ada.score(X_validation, y_validation)
    ada_test = ada.score(X_test, y_test)

    # Top Predictor
    ada_coef = ada.feature_importances_
    top_predictor = X_train.columns[np.argmax(np.abs(ada_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(ada_coef))
    ax.barh(X_train.columns[sorted_index], abs(ada_coef)[sorted_index])
    ax.set_title('AdaBoost Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['AdaBoost', ada_train, ada_validation, ada_test,
               (f"n_estimators={best_params['n_estimators']}, "
                f"learning_rate={best_params['learning_rate']}"),
               top_predictor, run_time]

    return results

def lgbm(X_train, y_train, X_validation, y_validation, X_test, y_test):
    # Hyperparamter tuning
    arr_max_depth = np.array([5, 7, 10])
    arr_subsample = np.array([0.7, 0.9])
    arr_learning_rate = np.array([0.01, 0.1])
    p = itertools.product(arr_max_depth, arr_subsample, arr_learning_rate)
    params = [i for i in p]

    validation_scores = []
    for param in tqdm(params):
        lgbm = ltb.LGBMRegressor(n_estimators=100, max_depth=param[0],
                                 subsample=param[1], learning_rate=param[2])
        lgbm.fit(X_train, y_train)
        validation_scores.append(lgbm.score(X_validation, y_validation))
    optim_params = params[np.argmax(validation_scores)]
    best_params = {'max_depth': optim_params[0],
                   'subsample': optim_params[1],
                   'learning_rate': optim_params[2]}

    # Start time
    start_time = time.time()

    # Training
    lgbm = ltb.LGBMRegressor(n_estimators=100,
                             max_depth=best_params['max_depth'],
                             subsample=best_params['subsample'],
                             learning_rate=best_params['learning_rate'])
    lgbm.fit(X_train, y_train)

    # End time
    run_time = (time.time() - start_time)

    # Scoring
    lgbm_train = lgbm.score(X_train, y_train)
    lgbm_validation = lgbm.score(X_validation, y_validation)
    lgbm_test = lgbm.score(X_test, y_test)

    # Top Predictor
    lgbm_coef = lgbm.feature_importances_
    top_predictor = X_train.columns[np.argmax(np.abs(lgbm_coef))]

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 15))
    sorted_index = np.argsort(abs(lgbm_coef))
    ax.barh(X_train.columns[sorted_index], abs(lgbm_coef)[sorted_index])
    ax.set_title('Light GBM Feature Importance Plot', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Summary results
    results = ['Light GBM', lgbm_train, lgbm_validation, lgbm_test,
               (f"max_depth={best_params['max_depth']}, "
                f"subsample={best_params['subsample']}, "
                f"learning_rate={best_params['learning_rate']}"),
               top_predictor, run_time]

    return results

a = lr(X_train, y_train, X_validation, y_validation, X_test, y_test)

# LR w/ Lasso
b = l1(X_train, y_train, X_validation, y_validation, X_test, y_test)

# LR w/ Ridge
c = l2(X_train, y_train, X_validation, y_validation, X_test, y_test)

# KNN
d = knn(X_train, y_train, X_validation, y_validation, X_test, y_test)

# Decision Tree
e = dt(X_train, y_train, X_validation, y_validation, X_test, y_test)

# Random Forest
f = rf(X_train, y_train, X_validation, y_validation, X_test, y_test)

# Gradient Boosting Method
g = gbm(X_train, y_train, X_validation, y_validation, X_test, y_test)

# XGBoost
h = xg(X_train, y_train, X_validation, y_validation, X_test, y_test)

# AdaBoost
i = ada(X_train, y_train, X_validation, y_validation, X_test, y_test)

# Light GBM
j = lgbm(X_train, y_train, X_validation, y_validation, X_test, y_test)

# Summary Table
cols = ['ML Method', 'Train Accuracy', 'Validaiton Accuracy', 'Test Accuracy',
        'Optimal Parameter', 'Top Predictor', 'Run Time']
df_summary = pd.DataFrame(columns=cols)
df_summary.loc[0] = a
df_summary.loc[1] = b
df_summary.loc[2] = c
df_summary.loc[3] = d
df_summary.loc[4] = e
df_summary.loc[5] = f
df_summary.loc[6] = g
df_summary.loc[7] = h
df_summary.loc[8] = i
df_summary.loc[9] = j
display(df_summary)

