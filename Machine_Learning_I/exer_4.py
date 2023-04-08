get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

fruits = pd.read_table('fruit_data_with_colors.txt')
display(fruits.head())
print(fruits.shape)

fruits.describe()

fruits.info()

fruits.isnull().sum()

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

Number_trials = 30

def train_knn(X, y):
    score_train = []
    score_test = []

    for seed in range(Number_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        neighbors_settings = range(1,31)
        acc_train = []
        acc_test = []

        for n_neighbors in neighbors_settings:   
            clf = KNeighborsClassifier(n_neighbors=n_neighbors) # build the model 
            clf.fit(X_train, y_train)    
            acc_train.append(clf.score(X_train, y_train))
            acc_test.append(clf.score(X_test, y_test))

        score_train.append(acc_train)
        score_test.append(acc_test)   
        
    score = np.mean(score_test, axis=0)
    run_time = (time.time() - start_time)
    return ['kNN', np.amax(score), 'N_Neighbor = {0}'.format(np.argmax(score)+1), 'NA',run_time]

def train_logistic(X, y, reg):
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    score_train = []
    score_test = []
    weighted_coefs=[]
    
    for seed in range(Number_trials):
        training_accuracy = []  
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        
        for alpha_run in C:
            if reg == 'l1':
                lr = LogisticRegression(C=alpha_run, penalty=reg, dual=False, solver='liblinear').fit(X_train, y_train)
            if reg == 'l2':
                lr = LogisticRegression(C=alpha_run, penalty=reg, dual=False).fit(X_train, y_train)
            training_accuracy.append(lr.score(X_train, y_train))
            test_accuracy.append(lr.score(X_test, y_test))
                
        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
    score = np.mean(score_test, axis=0)
    optimal_c = C[np.argmax(score)]
    
    score_train = []
    score_test = []
    for seed in range(Number_trials):
        training_accuracy = []  
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        
        if reg == 'l1':
            lr = LogisticRegression(C=optimal_c, penalty=reg, dual=False, solver='liblinear').fit(X_train, y_train)
        if reg == 'l2':
            lr = LogisticRegression(C=optimal_c, penalty=reg, dual=False).fit(X_train, y_train)
            
        training_accuracy.append(lr.score(X_train, y_train))
        test_accuracy.append(lr.score(X_test, y_test))
        weighted_coefs.append(lr.coef_) #append all the computed coefficients per trial
                
        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
    
    score = np.mean(score_test, axis=0)
    mean_coefs=np.mean(weighted_coefs, axis=0) #get the mean of the weighted coefficients over all the trials 
    #Plot the weight of the parameters 
    top_predictor= X.columns[np.argmax(np.max(np.abs(mean_coefs), axis=0))]
    coefs_count = len(mean_coefs)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.barh(np.arange(coefs_count), sorted(np.max(np.abs(mean_coefs), axis=0)))
    ax.set_yticks(np.arange(coefs_count))
    ax.set_yticklabels(X.columns[np.argsort(np.max(np.abs(mean_coefs), axis=0))])
    
    run_time = (time.time() - start_time)
    top_pred = {}
    for i, fruit in enumerate(fruits['fruit_name'].unique()):
        top_pred[fruit] = X.columns[np.argmax(np.max(np.abs(mean_coefs[i]), axis=0))]
    return ['Logistic ({0})'.format(reg), np.amax(score), 
            'C = {0}'.format(optimal_c), top_pred['apple'], top_pred['mandarin'],run_time]

def train_svm(X, y, reg):
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    score_train = []
    score_test = []
    weighted_coefs=[]
    
    for seed in range(Number_trials):
        training_accuracy = []  
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        
        for alpha_run in C:
            if reg == 'l1':
                svc = LinearSVC(C=alpha_run, penalty=reg, dual=False, loss='squared_hinge').fit(X_train, y_train)
            if reg == 'l2':
                svc = LinearSVC(C=alpha_run, penalty=reg).fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_test, y_test))
                
        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
    score = np.mean(score_test, axis=0)
    optimal_c = C[np.argmax(score)]
    
    score_train = []
    score_test = []
    for seed in range(Number_trials):
        training_accuracy = []  
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        
        if reg == 'l1':
            svc = LinearSVC(C=optimal_c, penalty=reg, dual=False, loss='squared_hinge').fit(X_train, y_train)
        if reg == 'l2':
            svc = LinearSVC(C=optimal_c, penalty=reg).fit(X_train, y_train)
            
        training_accuracy.append(svc.score(X_train, y_train))
        test_accuracy.append(svc.score(X_test, y_test))
        weighted_coefs.append(svc.coef_) #append all the computed coefficients per trial
                
        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
    
    score = np.mean(score_test, axis=0)
    mean_coefs=np.mean(weighted_coefs, axis=0) #get the mean of the weighted coefficients over all the trials 
    #Plot the weight of the parameters 
    top_predictor=X.columns[np.argmax(np.max(np.abs(mean_coefs), axis=0))]
    coefs_count = len(mean_coefs)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.barh(np.arange(coefs_count), sorted(np.max(np.abs(mean_coefs), axis=0)))
    ax.set_yticks(np.arange(coefs_count))
    ax.set_yticklabels(X.columns[np.argsort(np.max(np.abs(mean_coefs), axis=0))])
    
    run_time = (time.time() - start_time)
    return ['Linear SVM ({0})'.format(reg), np.amax(score), 
            'C = {0}'.format(optimal_c), top_predictor, run_time]

import time

start_time = time.time()
b = train_logistic(X,y,reg='l2')
print(b)
print("%s seconds" % b[4])

start_time = time.time()
c = train_logistic(X,y,reg='l1')
print(c)
print("%s seconds" % c[4])

cols = ['Machine Learning Method', 'Test Accuracy', 'Best Parameter', 'Top Predictor Variable Apple', 'Top Predictor Variable Mandarin', 'Top Predictor Variable Lemon','Top Predictor Variable Orange','Run Time']
df2 = pd.DataFrame(columns=cols)

df2.loc[1] = b
df2.loc[2] = c

