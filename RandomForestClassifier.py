#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
#导入数据，查看数据的分布类型
warnings.filterwarnings('ignore')
train = pd.read_csv('toxic_peptide/New Data/Dipepdite_Residue.csv')
target = 'target'
#train.info()
x_columns = [x for x in train.columns if x not in[target]]
X = train[x_columns]
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train.astype('int'))
rfc_y_predict = rfc.predict(X_test)
target_names = ['negative','positive']
print(rfc.score(X_test, y_test))
print(classification_report(y_test, rfc_y_predict, target_names=target_names))


# In[6]:


#对n_estimators进行网格搜索
#n_estimators值的确定
param_test1 = {'n_estimators':range(2,201,2)}
gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                 min_samples_leaf=20, max_depth=8, max_features='sqrt', random_state=10),
                       param_grid = param_test1, cv=5)
gsearch1.fit(X,y.astype('int'))
n_estimator = gsearch1.best_params_['n_estimators']
print('{} is : {}'.format('gsearch1_best_score', gsearch1.best_score_))

#max_depth、min_samples_split值的确定
param_test2= {'max_depth':range(2,21,2), 'min_samples_split':range(2,11,2)}
gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators=n_estimator,
                                 min_samples_leaf=20, max_features='sqrt' , oob_score=True, random_state=10),
   param_grid = param_test2, iid=False, cv=5)
gsearch2.fit(X,y.astype('int'))
max_depths = gsearch2.best_params_['max_depth']
min_samples_splits = gsearch2.best_params_['min_samples_split']
print('{} is: {}'.format('gsearch2_best_score',gsearch2.best_score_))

#min_samples_leaf、min_samples_split值的确定
param_test3= {'min_samples_leaf':range(2,11,2), 'min_samples_split':range(2,21,2)}
gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depths,
                                 max_features='sqrt', oob_score=True, random_state=10),
   param_grid = param_test3, iid=False, cv=5)
gsearch3.fit(X,y.astype('int'))
min_samples_leafs = gsearch3.best_params_['min_samples_leaf']
min_samples_splits = gsearch3.best_params_['min_samples_split']
print('{} is: {}'.format('gsearch3_best_score',gsearch3.best_score_))

#max_features值的确定
param_test4 = {'max_features':range(2,51,2)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depths, 
                            min_samples_split=min_samples_splits,min_samples_leaf=min_samples_leafs, oob_score=True, random_state=10),
   param_grid = param_test4, iid=False, cv=5)
gsearch4.fit(X,y.astype('int'))
max_feature = gsearch4.best_params_['max_features']
print('{} is: {}'.format('gsearch4_best_score',gsearch4.best_score_))

print('{}is:{},{}is:{},{}is:{},{}is:{},{}is:{}'.format('n_estimators',n_estimator,'max_depth',max_depths,
                             'min_samples_split',min_samples_splits,'min_samples_leaf',min_samples_leafs,'max_features',max_feature))

#调参前、后模型效果对比							 
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y.astype('int'))
print(rf0.oob_score_)

rf2 = RandomForestClassifier(n_estimators=140, max_depth=20, min_samples_split=10,
                                 min_samples_leaf=2, max_features=12, oob_score=True, random_state=10)
rf2.fit(X,y.astype('int'))
print(rf2.oob_score_)


# In[426]:

#确定模型各参数进行测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rfc1 = RandomForestClassifier(n_estimators=140, max_depth=20, min_samples_split=10,
                                 min_samples_leaf=2, max_features=12, oob_score=True, random_state=10)
rfc1.fit(X_train, y_train.astype('int'))
rfc1_y_predict = rfc1.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, rfc1_y_predict)
AUC = auc(fpr, tpr)
print("oob_score is: ", rfc1.oob_score_)
print("train_score is:", rfc1.score(X_train, y_train), " test_score is:", rfc1.score(X_test, y_test))
target_names=["NB","HB"]
print(classification_report(y_test, rfc1_y_predict, target_names=target_names), AUC)

