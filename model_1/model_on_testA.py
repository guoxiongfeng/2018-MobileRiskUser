#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import ensemble

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb

import msg_process
import voice_process
import wa_process

import helper
from helper import *


def xgb_evalMetric(preds,dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds=pre.preds.map(lambda x: 1 if x>=0.3 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc +0.4*f1
    return 'res',res



def lgb_evalMetric(preds,dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds=pre.preds.map(lambda x: 1 if x>=0.3 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc +0.4*f1
    return 'res',res, True



msg = msg_process.get_msg_feature_matrix()
tel = voice_process.get_voice_feature_matrix()
wa = wa_process.get_wa_feature_matrix()


smg_temp = msg.copy(deep=True)
tel_temp = tel.copy(deep=True)
wa_temp = wa.copy(deep=True)


smg_temp.set_index('uid',inplace=True)
tel_temp.set_index('uid', inplace=True)
wa_temp.set_index('uid', inplace=True)

summary = pd.concat([smg_temp, tel_temp, wa_temp],axis = 1)
#summary.head()

summary.fillna(0, inplace=True)

summary.index.name = 'uid'
summary.reset_index(inplace=True)


X_train, X_test, y_train = feature_matrix_to_train_matrix(summary)


#xgb
dtrain = xgb.DMatrix(X_train,label=y_train)


xgb_models = []
for i in range(10):
    xgb_params = {
        'booster':'gbtree',
        'objective':'binary:logistic',
        'stratified':True,
        'max_depth':20,
         'gamma':0,
        'subsample':0.8,
        'colsample_bytree':0.8,
        #'lambda':0.5,
        'eta':0.08,
        'seed':7 * i,
        'min_child_weight': 20,
        'silent':1
    }
    xgb_model=xgb.train(xgb_params,dtrain=dtrain,num_boost_round=300,verbose_eval=10,evals=[(dtrain,'train')],
                        feval=xgb_evalMetric,early_stopping_rounds=50, maximize=True)
    xgb_models.append(xgb_model)


xgb_gather = []
xgb_dtest = xgb.DMatrix(X_test)
for model in xgb_models:
    xgb_x = model.predict(xgb_dtest)
    xgb_gather.append(xgb_x)


xgb_pred = list(pd.DataFrame(xgb_gather).mean())


#lgb
dtrain = lgb.Dataset(X_train,label=y_train)
dtest = lgb.Dataset(X_test)

lgb_models = []
for i in range(10):
    lgb_params =  {
        'boosting_type': 'gbdt',
        'objective': 'binary',
    #    'metric': ('multi_logloss', 'multi_error'),
        #'metric_freq': 100,
        'is_training_metric': False,
        'min_data_in_leaf': 20,
        'num_leaves': 16,
        'learning_rate': 0.12,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity':-1,
       'feature_fraction_seed': i,
    #    'gpu_device_id':2,
    #    'device':'gpu'
       #'lambda_l1': 10,
    #    'skip_drop': 0.95,
    #    'max_drop' : 10
       # 'lambda_l2': 50
        #'num_threads': 18
    }    
    lgb_model =lgb.train(lgb_params,dtrain,feval=lgb_evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain], early_stopping_rounds= 50)
    lgb_models.append(lgb_model)


lgb_gather = []
for model in lgb_models:
    lgb_x = model.predict(X_test)
    lgb_gather.append(lgb_x)

lgb_pred = list(pd.DataFrame(lgb_gather).mean())



#lgb xgb 模型融合
lgb_xgb_pred = pd.DataFrame(list(map(lambda x, y, z: [x, y, z], label_test.uid, xgb_pred, lgb_pred)))
lgb_xgb_pred.columns = ['uid', 'xgb', 'lgb']


lgb_xgb_pred['sum'] = lgb_xgb_pred['xgb'] +  lgb_xgb_pred['lgb']

lgb_xgb_pred=lgb_xgb_pred.sort_values(by='sum',ascending=False)

lgb_xgb_pred['label']=lgb_xgb_pred['sum'].map(lambda x: 1 if x>= 0.9 else 0)

lgb_xgb_pred['label'] = lgb_xgb_pred['label'].map(lambda x: int(x))

lgb_xgb_pred.to_csv('../result/test_A_res.csv',index=False,header=False,sep=',',columns=['uid','label'])













