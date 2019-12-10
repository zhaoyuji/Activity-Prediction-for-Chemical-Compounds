"""
Step 3: Get the final prediction on given testing set without labels
    Based on the results before, we found that when Mol2vec features and LightGBM
    are used we got the highest estimated AUC 0.810312571. 
    We used the whole training set and best parameters to train the final model
    and got the final prediction on given testing set
Input: 
    "traindict.p"
    "testdict.p"

Output:
    predition.txt
    The first line is our estimated AUC andthe following is our prediction results of 6375 testing samples
    
"""

import warnings
warnings.filterwarnings('ignore')#处理警告

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
import lightgbm as lgb



traindict = pickle.load(open( "traindict.p", "rb" ) )
testdict = pickle.load(open( "testdict.p", "rb" ) )

train_x = traindict['molvec']
train_y = np.array(traindict['y'].astype(int))  

test_x = testdict['molvec']



print("LightGBM:")
best_params={'bagging_fraction': 0.8, 'bagging_freq': 2, 'cat_smooth': 10, 
             'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.03, 
             'max_depth': -1, 'min_data_in_leaf': 40, 'min_sum_hessian_in_leaf': 6, 
             'num_leaves': 60, 'metric': 'auc','objective': 'binary',
             'random_state':2019, 'boosting_type':'gbdt','verbose':-1
             #'verbose_eval':100
             }

clf = lgb.train(best_params,
                lgb.Dataset(train_x, train_y),
                #early_stopping_rounds=10,
                num_boost_round=500)

print("training AUC: {}".format(roc_auc_score(train_y, clf.predict(train_x))))

# prediction
proba = clf.predict(test_x, num_iteration=clf.best_iteration)
output = np.concatenate(([0.810312571], proba)) 
np.savetxt('predition.txt', output)
           