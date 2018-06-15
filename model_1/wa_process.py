import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBRegressor

#Note: 每一轮的Test是不同的uid！ 记得修改这里！
label_test = pd.DataFrame(['u' + str(x + 5000) for x in range(2000)], columns=['uid'])
label_train = pd.read_csv("../data/train/uid_train.txt",header=None,names=['uid','label'], delimiter= '\t')

def read_data():
    names_wa = ['uid', 'wa_name','visit_cnt','visit_dura','up_flow', 'down_flow', 'wa_type', 'wa_date']
    wa_train = pd.read_csv("../data/train/wa_train.txt",header=None,encoding='utf-8',names = names_wa,index_col = False, delimiter='\t')
    wa_test = pd.read_csv("../data/test/wa_test_a.txt",header=None,encoding='utf-8',names = names_wa,index_col = False, delimiter='\t')
    wa_data = pd.concat([wa_train, wa_test])
    return wa_data



def get_wa_name_feature(wa_data):
    wa_name_unique_cnt = wa_data.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('wa_name_')
    wa_name_unique_cnt_web = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('wa_name_web_')
    wa_name_unique_cnt_app = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('wa_name_app_')
    wa_name_unique_cnt = pd.concat([wa_name_unique_cnt, wa_name_unique_cnt_web, wa_name_unique_cnt_app], axis = 1)
    return wa_name_unique_cnt

def get_wa_name_deep_feature(wa_data):
    #按照坏的app构建特征
    tmp0 = pd.DataFrame(wa_data[(wa_data['uid'] >= 'u4100') & (wa_data['uid'] < 'u5000')].groupby(['wa_name', 'uid'])['wa_name'].count().groupby(['wa_name']).count())
    tmp1 = pd.DataFrame(wa_data[wa_data['uid'] < 'u4100'].groupby(['wa_name', 'uid'])['wa_name'].count().groupby(['wa_name']).count())
    tmp = pd.concat([tmp0,tmp1],axis = 1)
    tmp.columns = ['bad_times', 'good_times']
    #tmp.dropna(axis = 0, inplace = True, )
    tmp.fillna(0, inplace = True)
    tmp['bad_ratio'] = tmp['bad_times'] / 900
    tmp['good_ratio'] = tmp['good_times'] / 4100
    tmp['diff'] = tmp['bad_ratio'] - tmp['good_ratio']
    #调节这里改变模型
    tmp[(tmp['bad_times'] >= 35) & (tmp['good_times'] <= 70)].sort_values(by='diff', ascending=False)
    bad_wa_names = list(tmp[(tmp['bad_times'] >= 35) & (tmp['good_times'] <= 70)].index)
    
    #bad_wa_names = ['viber', '中国电信网上营业厅', '卡牛信用卡管家', '友店', '哆啦宝', '微众', '本地宝导航', '每日优鲜', '环球黑卡']

    bad_wa_names_cnt = []
    for name in bad_wa_names:
        bad_wa_names_cnt.append(wa_data[wa_data['wa_name'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('bad_wa_names_' + name + '_cnt_'))
    bad_wa_names_cnt = pd.concat(bad_wa_names_cnt, axis = 1)
    bad_wa_names_cnt.fillna(0, inplace=True)


    #按照热门的app构建特征
    t = pd.DataFrame(wa_data['wa_name'].value_counts())
    #调节这里改变模型
    hot_wa_names = list(t[t['wa_name'] > 50000].index)

    

    hot_wa_names_cnt = []
    for  name in hot_wa_names:
        hot_wa_names_cnt.append(wa_data[wa_data['wa_name'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('hot_wa_names_' + name + '_cnt_'))
    hot_wa_names_cnt = pd.concat(hot_wa_names_cnt, axis = 1)
    hot_wa_names_cnt.fillna(0, inplace=True)

    wa_names_deep = pd.concat([bad_wa_names_cnt, hot_wa_names_cnt], axis = 1)
    
    return wa_names_deep

def get_wa_date_category_feature(wa_data):
    #统计按照时间bag_size划分的流量总量(占比)
    date_bag_size = 5
    wa_date_category_flow_amount = wa_data.groupby(['uid'])['down_flow'].agg(['sum'])
    wa_date_category_flow = []
    for cat_num in range(45 // date_bag_size):
        wa_date_category_flow.append((wa_data[(wa_data['wa_date'] - 1) // date_bag_size == cat_num].groupby(
            ['uid'])['down_flow'].agg(['sum']) / wa_date_category_flow_amount).add_prefix('wa_date_cat_flow_' + str(cat_num) + '_'))
    wa_date_category_flow = pd.concat(wa_date_category_flow, axis = 1)
    #wa_date_category_flow.head(10)

    #统计按照时间bag_size划分的流量使用次数(占比)
    wa_date_category_cnt_amount = wa_data.groupby(['uid'])['uid'].agg(['count'])
    wa_date_category_cnt = []
    for cat_num in range(45 // date_bag_size):
        wa_date_category_cnt.append((wa_data[(wa_data['wa_date'] - 1) // date_bag_size == cat_num].groupby(
            ['uid'])['uid'].agg(['count']) / wa_date_category_cnt_amount).add_prefix('wa_date_cat_' + str(cat_num) + '_') )
    wa_date_category_cnt = pd.concat(wa_date_category_cnt, axis = 1)

    #统计按照时间bag_size划分的使用时间(占比)
    wa_date_category_dura_amount = wa_data.groupby(['uid'])['visit_dura'].agg(['sum'])
    wa_date_category_dura = []
    for cat_num in range(45 // date_bag_size):
        wa_date_category_dura.append((wa_data[(wa_data['wa_date'] - 1) // date_bag_size == cat_num].groupby(
            ['uid'])['visit_dura'].agg(['sum']) / wa_date_category_cnt_amount).add_prefix('wa_date_cat_dura_' + str(cat_num) + '_') )
    wa_date_category_dura = pd.concat(wa_date_category_dura, axis = 1)

    wa_date_category = pd.concat([wa_date_category_flow, wa_date_category_cnt, wa_date_category_dura], axis = 1)


def get_wa_cnt_feature(wa_data):
    wa_cnt = wa_data.groupby(['uid'])['wa_name'].agg(['count']).add_prefix('wa_')
    wa_cnt_web = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['wa_name'].agg(['count']).add_prefix('wa_').add_prefix('wa_web_')
    wa_cnt_app = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['wa_name'].agg(['count']).add_prefix('wa_').add_prefix('wa_app_')
    wa_cnt = pd.concat([wa_cnt, wa_cnt_web, wa_cnt_app], axis = 1)
    wa_cnt.fillna(0,inplace=True)
    wa_cnt['wa_cnt_diff'] = wa_cnt['wa_web_wa_count'] - wa_cnt['wa_app_wa_count']
    return wa_cnt


def get_wa_visit_cnt_feature(wa_data):
    wa_visit_cnt_date_static = wa_data.groupby(['uid', 'wa_date'])['visit_cnt'].sum().groupby(['uid']).agg(
    ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_visit_cnt_date_')
    wa_web_visit_cnt_date_static = wa_data[wa_data['wa_type'] == 0].groupby(['uid', 'wa_date'])['visit_cnt'].sum().groupby(['uid']).agg(
    ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_web_visit_cnt_date_')
    wa_app_visit_cnt_date_static = wa_data[wa_data['wa_type'] == 1].groupby(['uid', 'wa_date'])['visit_cnt'].sum().groupby(['uid']).agg(
    ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_app_visit_cnt_date_')
    wa_visit_cnt_date_static = pd.concat([wa_visit_cnt_date_static, wa_web_visit_cnt_date_static, wa_app_visit_cnt_date_static], axis = 1)
    wa_visit_cnt_date_static.fillna(0, inplace=True)
    wa_visit_cnt_date_static['wa_visit_vnt_date_diff'] = wa_visit_cnt_date_static['wa_web_visit_cnt_date_sum'] - wa_visit_cnt_date_static['wa_app_visit_cnt_date_sum']


    #wa_visit_cnt_date_static.head()
    wa_visit_cnt_static = wa_data.groupby(['uid'])['visit_cnt'].agg(['std','max','median','mean','sum']).add_prefix('wa_visit_cnt_')
    wa_web_visit_cnt_static = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['visit_cnt'].agg(['std','max','median','mean','sum']).add_prefix('wa_web_visit_cnt_')
    wa_app_visit_cnt_static = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['visit_cnt'].agg(['std','max','median','mean','sum']).add_prefix('wa_app_visit_cnt_')

    wa_visit_cnt_static = pd.concat([wa_visit_cnt_static, wa_web_visit_cnt_static, wa_app_visit_cnt_static], axis = 1)
    wa_visit_cnt_static.fillna(0, inplace= True)
    wa_visit_cnt_static['wa_visit_cnt_diff'] = wa_visit_cnt_static['wa_web_visit_cnt_sum'] - wa_visit_cnt_static['wa_web_visit_cnt_sum']


    return pd.concat([wa_visit_cnt_date_static, wa_visit_cnt_static], axis = 1)

def get_wa_visit_dura_feature(wa_data):
    wa_visit_dura_date_static = wa_data.groupby(['uid', 'wa_date'])['visit_dura'].sum().groupby(['uid']).agg(
        ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_visit_dura_date_')
    wa_visit_web_dura_date_static = wa_data[wa_data['wa_type'] == 0].groupby(['uid', 'wa_date'])['visit_dura'].sum().groupby(['uid']).agg(
        ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_visit_web_dura_date_')
    wa_visit_app_dura_date_static = wa_data[wa_data['wa_type'] == 1].groupby(['uid', 'wa_date'])['visit_dura'].sum().groupby(['uid']).agg(
        ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_visit_app_dura_date_')
    wa_visit_dura_date_static = pd.concat([wa_visit_dura_date_static, wa_visit_web_dura_date_static, wa_visit_app_dura_date_static], axis = 1)
    wa_visit_dura_date_static.fillna(0, inplace=True)

    wa_visit_dura_date_static['wa_visit_dura_date_diff'] = wa_visit_dura_date_static['wa_visit_web_dura_date_sum'] - wa_visit_dura_date_static['wa_visit_app_dura_date_sum']

    return wa_visit_dura_date_static

def get_visit_dura_cat(wa_data):
    wa_visit_dura_cat_0 = wa_data[wa_data['visit_dura'] < 10].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_1 = wa_data[(wa_data['visit_dura'] >= 10) & (wa_data['visit_dura'] < 1e2)].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_2 = wa_data[(wa_data['visit_dura'] >= 1e2) & (wa_data['visit_dura'] < 1e3)].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_3 = wa_data[(wa_data['visit_dura'] >= 1e3) & (wa_data['visit_dura'] < 1e4)].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_4 = wa_data[(wa_data['visit_dura'] >= 1e4) & (wa_data['visit_dura'] < 1e5)].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_5 = wa_data[(wa_data['visit_dura'] >= 1e5) & (wa_data['visit_dura'] < 1e5)].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_6 = wa_data[(wa_data['visit_dura'] >= 1e6) & (wa_data['visit_dura'] < 1e7)].groupby(['uid'])['uid'].count()
    wa_visit_dura_cat_7 = wa_data[wa_data['visit_dura'] >= 1e7].groupby(['uid'])['uid'].count()

    wa_visit_dura_cat = pd.concat([wa_visit_dura_cat_0, wa_visit_dura_cat_1,wa_visit_dura_cat_2, wa_visit_dura_cat_3, 
                                wa_visit_dura_cat_4, wa_visit_dura_cat_5, wa_visit_dura_cat_6, wa_visit_dura_cat_7], axis = 1)
    wa_visit_dura_cat.columns = ['wa_visit_dura_cat_0', 'wa_visit_dura_cat_1','wa_visit_dura_cat_2', 'wa_visit_dura_cat_3', 
                                'wa_visit_dura_cat_4', 'wa_visit_dura_cat_5', 'wa_visit_dura_cat_6', 'wa_visit_dura_cat_7']
    wa_visit_dura_cat.fillna(0, inplace=True)
    return wa_visit_dura_cat


def get_up_down_flow_feature(wa_data):
    up_flow_static = wa_data.groupby(['uid'])['up_flow'].agg(['std','max','median','mean','sum']).add_prefix('wa_up_flow_')
    up_flow_web_static = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['up_flow'].agg(['std','max','median','mean','sum']).add_prefix('wa_web_up_flow_')
    up_flow_app_static = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['up_flow'].agg(['std','max','median','mean','sum']).add_prefix('wa_app_up_flow_')
    up_flow_static = pd.concat([up_flow_static, up_flow_web_static, up_flow_app_static], axis = 1)

    down_flow_static = wa_data.groupby(['uid'])['down_flow'].agg(['std','max','median','mean','sum']).add_prefix('wa_down_flow_')
    down_flow_web_static = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['down_flow'].agg(['std','max','median','mean','sum']).add_prefix('wa_web_down_flow_')
    down_flow_app_static = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['down_flow'].agg(['std','max','median','mean','sum']).add_prefix('wa_app_down_flow_')
    down_flow_static = pd.concat([down_flow_static, down_flow_web_static, down_flow_app_static], axis = 1)
    
    flow_static = wa_data.groupby(['uid'])['flow_amount'].agg(['std','max','median','mean','sum']).add_prefix('wa_amount_flow_')
    flow_web_static = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['flow_amount'].agg(['std','max','median','mean','sum']).add_prefix('wa_web_amount_flow_')
    flow_app_static = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['flow_amount'].agg(['std','max','median','mean','sum']).add_prefix('wa_app_amount_flow_')
    flow_static = pd.concat([flow_static, flow_web_static, flow_app_static], axis = 1)

    flow_static = pd.concat([up_flow_static, down_flow_static, flow_static],axis = 1)


    return flow_static

def get_other_feature(wa_data):
    wa_visit_day_cnt = wa_data.groupby(['uid', 'wa_date'])['wa_date'].count().groupby(['uid']).count() #有几天使用了流量

    wa_data['visit_per_dura'] = wa_data['visit_dura'] / wa_data['visit_cnt']
    wa_data['flow_amount'] = wa_data['up_flow'] + wa_data['down_flow']

    wa_visit_per_dura_static = wa_data.groupby(['uid', 'wa_date'])['visit_per_dura'].sum().groupby(['uid']).agg(
    ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_visit_per_dura_')

    wa_data['download_speed'] = (wa_data['down_flow'] + 1) / (wa_data['visit_dura'] + 1)
    wa_data['upload_speed'] = (wa_data['up_flow'] + 1) / (wa_data['visit_dura'] + 1)
    wa_data['amount_speed'] = wa_data['download_speed'] + wa_data['upload_speed']

    wa_download_speed_static = wa_data.groupby(['uid'])['download_speed'].agg(['std','max','median','mean','sum']).add_prefix('download_speed_')
    wa_upload_speed_static = wa_data.groupby(['uid'])['upload_speed'].agg(['std','max','median','mean','sum']).add_prefix('upload_speed_')
    wa_amount_speed_static = wa_data.groupby(['uid'])['amount_speed'].agg(['std','max','median','mean','sum']).add_prefix('amount_speed_')

    wa_speed_static = pd.concat([wa_download_speed_static, wa_upload_speed_static, wa_amount_speed_static], axis = 1)

    others = pd.concat([wa_visit_day_cnt, wa_visit_per_dura_static, wa_speed_static], axis = 1)

    return others


def get_wa_feature_matrix():
    wa_data = read_data()
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

    return wa
    