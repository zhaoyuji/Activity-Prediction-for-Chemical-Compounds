"""
Step 2: Tune Parameters on validation set and get estimated AUC on testing set

Input: 
    traindict.p

Output:
    print resultsof estimated AUC and best parameters
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


traindict = pickle.load(open( "traindict.p", "rb" ) )
train_y = traindict['y']
traindict = {"Morgan": traindict['Morgan'], 
             "Despcritor": traindict['Despcritor'],
             "molvec": traindict['molvec']}

def split_dataset(traindict, train_y, feature=["Morgan"], ratio=0.8):
    data = pd.DataFrame()
    for f in features:
        data = pd.concat([data, traindict[f]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, train_y, test_size=ratio, random_state=10, stratify=train_y) # Stratified sampling
    return X_train, X_test, y_train, y_test

features = ["Morgan", "Despcritor", "molvec"] # choose which features to use
trainx, testx, trainy, testy = split_dataset(traindict, train_y, feature=features, ratio=0.2)

## use gridsearch cv on training set to tune parameters
X = np.array(trainx) 
y = np.array(trainy.astype(int))  

print("Samples number: all, train, val, test: {}, {}, {}, {}".format(len(train_y), int(4*len(trainx)/5), int(len(trainx)/5), len(testy))) 
print()

########################### LogisticRegression ###########################
print("LogisticRegression: \n")

best_ting = {
    'max_iter': [60], #[60, 100],
    'C':[0.1], #[0.1,1],
    'class_weight': [None]#["balanced", None]
}

best_g = GridSearchCV(LogisticRegression(), best_ting, scoring="roc_auc", verbose=1, cv=5)
best_g.fit(X, y)
print(best_g.best_params_)#输出最优参数
print("Validation set best AUC mean: {}".format(best_g.best_score_))

best_model = best_g.best_estimator_
best_model.fit(trainx, trainy)
probs = best_model.predict_proba(testx)
print("Test set AUC: {}".format(roc_auc_score(testy,probs[:,-1])))

print()

########################### RandomForest ###########################
print("Random Forest: \n")

parameter_space = {
    "n_estimators": [20], #[20, 50, 100],
    "min_samples_leaf": [10] #[6, 10, 30],
}
 
clf = RandomForestClassifier(random_state=14, criterion= "gini")
grid = GridSearchCV(clf, parameter_space, cv=5, scoring='roc_auc')
grid.fit(X, y)
print(grid.best_params_)#输出最优参数
print("Validation set best AUC mean: {}".format(grid.best_score_))


best_model = grid.best_estimator_
best_model.fit(trainx, trainy)
probs = best_model.predict_proba(testx)
print("Test set AUC: {}".format(roc_auc_score(testy,probs[:,-1])))

print()


########################### LightGBM ###########################
print("LightGBM:")

parameters = {
            'num_leaves': [60], #[60,100], 
            'min_data_in_leaf': [40],
            'max_depth': [-1],  #[15, 20, 25, 30, 35],
            "min_sum_hessian_in_leaf": [6],
            'learning_rate': [0.01],  #[0.01, 0.02, 0.05, 0.1, 0.15],
            'feature_fraction': [0.9],  #[0.6, 0.7, 0.8, 0.9, 0.95],
            'bagging_fraction': [0.8],
            'bagging_freq': [2],
#            "bagging_seed": [11],
            'lambda_l1': [0.1],
#            'lambda_l2': [0],
            'cat_smooth': [10]
}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
#                         verbose=100,
                         verbose=-1,
                         num_boost_round=500,
                         random_state = 2019
                         )

gsearch = GridSearchCV(gbm, param_grid=parameters, scoring="roc_auc", verbose=1, cv=5)
gsearch.fit(X, y)
print(gsearch.best_params_)
print("Validation set best AUC mean: {}".format(gsearch.best_score_))


best_model = gsearch.best_estimator_
best_model.fit(trainx, trainy)
probs = best_model.predict_proba(testx)
print("Test set AUC: {}".format(roc_auc_score(testy,probs[:,-1])))








