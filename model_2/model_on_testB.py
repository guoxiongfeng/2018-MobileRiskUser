#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBRegressor

import lightgbm as lgb

import msg_process
import voice_process
import wa_process

from msg_process import *
from voice_process import *
from wa_process import *

import importlib as imp
imp.reload(msg_process)
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


#Note: 每一轮的Test是不同的uid！ 记得修改这里！
label_test = pd.DataFrame(['u' + str(x + 7000) for x in range(3000)], columns=['uid'])
label_train = pd.read_csv("../data/train/uid_train.txt",header=None,names=['uid','label'], delimiter= '\t')
label_train_a = pd.read_csv("../result/test_A_res.csv",header=None,names=['uid','label'], delimiter= ',')
label_train = pd.concat([label_train, label_train_a])



#msg
names_smg = ['uid','smg_opp_num','smg_opp_head','smg_opp_len','smg_start_time','smg_in_and_out']
smg_train = pd.read_csv("../data/train/sms_train.txt",header=None,encoding='utf-8',names = names_smg,index_col = False, delimiter='\t',
                    dtype= {'smg_start_time': str,
                               'smg_opp_head': str,
                               'smg_opp_len': int,
                               'smg_in_and_out': int
                            })
smg_train_a = pd.read_csv("../data/test/sms_test_a.txt",header=None,encoding='utf-8',names = names_smg,index_col = False, delimiter='\t',
                    dtype= {'smg_start_time': str,
                            'smg_opp_head': str,
                            'smg_opp_len': int,
                            'smg_in_and_out': int
                            })


smg_test = pd.read_csv("../data/test/sms_test_b.txt",header=None,encoding='utf-8',names = names_smg,index_col = False, delimiter='\t',
                    dtype= {'smg_start_time': str,
                            'smg_opp_head': str,
                            'smg_opp_len': int,
                            'smg_in_and_out': int
                            })
msg_data = pd.concat([smg_train, smg_train_a, smg_test])
msg_data['smg_hour'] = msg_data['smg_start_time'].apply(lambda x: int(x[2:4]))
msg_data['smg_date'] = msg_data['smg_start_time'].apply(lambda x: int(x[0:2]))

#copy from get matrix method.
smg_opp_len_div = get_msg_opp_len_div(msg_data)
smg_in_and_out = get_msg_in_and_out(msg_data)
foreign_fre = get_smg_foreign_fre(msg_data)
msg_operator_div = get_msg_operator_div(msg_data)
others = get_smg_other_feature(msg_data)
msg_opp_head_catagory = get_msg_opp_head_catagory(msg_data)
msg_opp_len_catagory = get_msg_opp_len_catagory(msg_data)
#先不使用hour信息，暂时用上会使正确率下降
msg_hour_feature = get_msg_hour_feature(msg_data)
msg_date_feature = get_msg_date_feature(msg_data)
msg_opp_num_feature = get_msg_opp_num_feature(msg_data)
smg = pd.concat([smg_opp_len_div, smg_in_and_out, foreign_fre, msg_operator_div, others, 
    msg_opp_head_catagory, msg_opp_len_catagory, msg_date_feature],axis =1)
smg['smg_fre'] = smg['smg_in_cnt'] + smg['smg_out_cnt']
smg['home_fre'] = smg['smg_fre'] - smg['foreign_fre'] #计算国内smg量
smg.fillna(0, inplace=True)
smg = smg.apply(lambda x: x.astype(int))
smg.index.name = 'uid'
smg.reset_index(inplace=True)




#Voice

names_voice = ['uid', 'tel_opp_num','tel_opp_head', 'tel_opp_len', 'tel_start_time', 'tel_end_time', 'tel_call_type', 'tel_in_and_out']
voice_train = pd.read_csv("../data/train/voice_train.txt",header=None,encoding='utf-8',names = names_voice,index_col = False, delimiter='\t', 
                        dtype = {
                            'tel_start_time': int,
                            'tel_opp_num':str,
                            'tel_opp_head':str
                        },
                        low_memory=False)

voice_train_a = pd.read_csv("../data/test/voice_test_a.txt",header=None,encoding='utf-8',names = names_voice,index_col = False, delimiter='\t', 
                        dtype = {
                            'tel_start_time': int,
                            'tel_opp_num':str,
                            'tel_opp_head':str
                        },
                        low_memory=False)

voice_test = pd.read_csv("../data/test/voice_test_b.txt",header=None,encoding='utf-8',names = names_voice,index_col = False, delimiter='\t', 
                        dtype = {
                            'tel_start_time': int,
                            'tel_opp_num':str,
                            'tel_opp_head':str
                        },
                        low_memory=False)
voice_data = pd.concat([voice_train, voice_train_a, voice_test])

    #计算通话量
    #转换出通话时间量
voice_data['tel_len'] = (((voice_data['tel_end_time'] //1e6 - voice_data['tel_start_time'] //1e6)  * 24 + \
(voice_data['tel_end_time'] //1e4 % 100 - voice_data['tel_start_time'] //1e4 % 100)) * 60 + \
(voice_data['tel_end_time'] //1e2 % 100 - voice_data['tel_start_time'] //1e2 % 100)) * 60 + \
(voice_data['tel_end_time'] % 100 - voice_data['tel_start_time'] % 100)
voice_data['tel_len'] = voice_data['tel_len'].astype(int)
    #转换出日期
voice_data['tel_date'] = voice_data['tel_start_time'].apply(lambda x: int(x // 1e6))
voice_data['tel_hour'] = voice_data['tel_start_time'].apply(lambda x: int(x // 1e4 % 100))



tel_in_and_out = get_tel_in_and_out(voice_data)
tel_type_category = get_tel_type_catagory(voice_data)

tel_len_mean = get_tel_len_mean(voice_data)
tel_len_sum = get_tel_len_sum(voice_data)
tel_len_max = get_tel_len_max(voice_data)
tel_len_div = get_tel_len_div(voice_data)

tel_opp_len_div = get_tel_opp_len_div(voice_data)

tel_date_feature = get_tel_date_feature(voice_data)
tel_opp_head_catagory = get_tel_opp_head_catagory(voice_data)
tel_opp_len_catagory = get_tel_opp_len_catagory(voice_data)
tel_hour_feature = get_tel_hour_feature(voice_data)
tel_opp_num_feature = get_tel_opp_num_feature(voice_data)
others = get_tel_other_feature(voice_data)

tel = pd.concat([tel_in_and_out, tel_type_category, tel_len_mean, tel_len_sum, tel_len_max, tel_len_div, tel_opp_len_div, others,
    tel_date_feature, tel_opp_head_catagory, tel_opp_len_catagory, tel_hour_feature], axis = 1)
tel.index.name = 'uid'
tel.reset_index(inplace=True)


#wa
names_wa = ['uid', 'wa_name','visit_cnt','visit_dura','up_flow', 'down_flow', 'wa_type', 'wa_date']
wa_train = pd.read_csv("../data/train/wa_train.txt",header=None,encoding='utf-8',names = names_wa,index_col = False, delimiter='\t')
wa_train_a = pd.read_csv("../data/test/wa_test_a.txt",header=None,encoding='utf-8',names = names_wa,index_col = False, delimiter='\t')
wa_test = pd.read_csv("../data/test/wa_test_b.txt",header=None,encoding='utf-8',names = names_wa,index_col = False, delimiter='\t')
wa_data = pd.concat([wa_train, wa_train_a, wa_test])


other_feature = get_other_feature(wa_data)
wa_name_feature = get_wa_name_feature(wa_data)
wa_name_deep_feature = get_wa_name_deep_feature(wa_data)
wa_cnt_feature = get_wa_cnt_feature(wa_data)
wa_visit_cnt_feature = get_wa_visit_cnt_feature(wa_data)
wa_visit_dura_feature = get_wa_visit_dura_feature(wa_data)
up_down_flow_feature = get_up_down_flow_feature(wa_data)
wa_date_category = get_wa_date_category_feature(wa_data)
wa_visit_dura_cat = get_visit_dura_cat(wa_data)
    
wa = pd.concat([wa_name_feature, wa_name_deep_feature, wa_cnt_feature, wa_visit_cnt_feature, 
wa_visit_dura_feature, wa_visit_dura_cat, up_down_flow_feature, wa_date_category, other_feature], axis = 1)
wa.index.name = 'uid'
wa.reset_index(inplace=True)



#合并

smg_temp = smg.copy(deep=True)
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



#xgboost
X_train, X_test, y_train = feature_matrix_to_train_matrix(summary)
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
        'eta':0.05,
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
        'is_training_metric': False,
        'min_data_in_leaf': 20,
        'num_leaves': 16,
        'learning_rate': 0.12,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity':-1,
       'feature_fraction_seed': i
    }    
    lgb_model =lgb.train(lgb_params,dtrain,feval=lgb_evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain], early_stopping_rounds= 50)
    lgb_models.append(lgb_model)



lgb_gather = []
for model in lgb_models:
    lgb_x = model.predict(X_test)
    lgb_gather.append(lgb_x)

lgb_pred = list(pd.DataFrame(lgb_gather).mean())


#xgb lgb 融合
lgb_xgb_pred = pd.DataFrame(list(map(lambda x, y, z: [x, y, z], label_test.uid, xgb_pred, lgb_pred)))
lgb_xgb_pred.columns = ['uid', 'xgb', 'lgb']

lgb_xgb_pred['sum'] = 0.96 * lgb_xgb_pred['xgb'] +  1.04 * lgb_xgb_pred['lgb']


lgb_xgb_pred=lgb_xgb_pred.sort_values(by='sum',ascending=False)

lgb_xgb_pred['label']=lgb_xgb_pred['sum'].map(lambda x: 1 if x>= 0.911 else 0)

lgb_xgb_pred['label'] = lgb_xgb_pred['label'].map(lambda x: int(x))

lgb_xgb_pred.to_csv('../result/test_B_res.csv',index=False,header=False,sep=',',columns=['uid','label'])

























