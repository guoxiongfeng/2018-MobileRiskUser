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


#目前看来XBG 最好  RF次之。
def tel_train_model(local_train_X, local_train_Y):
    #model = ensemble.RandomForestClassifier(n_estimators=200, criterion= 'gini', max_depth=None)
    #model = KNeighborsClassifier(n_neighbors=1)
    #model = MLPClassifier(hidden_layer_sizes = 100, tol = 1e-4)
    model = XGBRegressor(learning_rate=0.05, n_estimators=200,max_depth = 12,min_child_weight=11)
    #model = SVC(C=10, probability=True)
    model.fit(local_train_X, local_train_Y)
    return model




def read_data():
    #数据完整无na
    names_voice = ['uid', 'tel_opp_num','tel_opp_head', 'tel_opp_len', 'tel_start_time', 'tel_end_time', 'tel_call_type', 'tel_in_and_out']
    voice_train = pd.read_csv("../data/train/voice_train.txt",header=None,encoding='utf-8',names = names_voice,index_col = False, delimiter='\t', 
                          dtype = {
                              'tel_start_time': int,
                              'tel_opp_num':str,
                              'tel_opp_head':str
                          },
                          low_memory=False)
    voice_test = pd.read_csv("../data/test/voice_test_a.txt",header=None,encoding='utf-8',names = names_voice,index_col = False, delimiter='\t', 
                          dtype = {
                              'tel_start_time': int,
                              'tel_opp_num':str,
                              'tel_opp_head':str
                          },
                          low_memory=False)
    voice_data = pd.concat([voice_train, voice_test])

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
    return voice_data

def get_tel_in_and_out(voice_data):
    #in-and-out
    tel_in_and_out = voice_data.groupby(['uid', 'tel_in_and_out'])['tel_in_and_out'].count().unstack('tel_in_and_out')
    tel_in_and_out.columns=['tel_in','tel_out']
    tel_in_and_out.fillna(0, inplace=True)
    tel_in_and_out = tel_in_and_out.apply(lambda x: x.astype(int))
    tel_in_and_out['tel_fre'] = tel_in_and_out['tel_in'] + tel_in_and_out['tel_out']
    tel_in_and_out['tel_in_out_ratio'] = (tel_in_and_out['tel_out'] + 1) / (tel_in_and_out['tel_in'] + 1)
    #tel_in_and_out.head()
    return tel_in_and_out





#感觉date的特征tel和msg两张表处理的都不是很好
def get_tel_date_feature(voice_data):
    tel_date_static = voice_data.groupby(['uid', 'tel_date'])['uid'].count().groupby(['uid']).agg(
        ['mean', 'count', 'sum', 'std']).add_prefix('tel_date_')
    tel_in_date_static = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid', 'tel_date'])['uid'].count().groupby(['uid']).agg(
        ['mean', 'count', 'sum', 'std']).add_prefix('tel_in_date_')
    tel_out_date_static = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid', 'tel_date'])['uid'].count().groupby(['uid']).agg(
        ['mean', 'count', 'sum', 'std']).add_prefix('tel_out_date_')
    tel_date_static = pd.concat([tel_date_static, tel_in_date_static, tel_out_date_static], axis = 1)

    tel_date_static.fillna(0, inplace=True)
    tel_date_static['tel_date_in_out_diff_mean'] = tel_date_static['tel_in_date_mean'] - tel_date_static['tel_out_date_mean']
    tel_date_static['tel_date_in_out_diff_count'] = tel_date_static['tel_in_date_count'] - tel_date_static['tel_out_date_count']
    tel_date_static['tel_date_in_out_diff_sum'] = tel_date_static['tel_in_date_sum'] - tel_date_static['tel_out_date_sum']
    tel_date_static['tel_date_in_out_diff_std'] = tel_date_static['tel_in_date_std'] - tel_date_static['tel_out_date_std']


    date_bag_size = 5
    tel_date_cnt_category = []
    for cat_num in range(45 // date_bag_size):
        tel_date_cnt_category.append((voice_data[(voice_data['tel_date'] - 1) // date_bag_size == cat_num].groupby(
            ['uid'])['uid'].agg(['count'])).add_prefix('tel_date_cnt_cat_' + str(cat_num) + '_') )
    tel_date_cnt_category = pd.concat(tel_date_cnt_category, axis = 1)

    tel_date_all_kinds = voice_data.groupby(['uid', 'tel_date'])['tel_date'].count().unstack('tel_date').add_prefix('tel_date_')
    tel_in_date_all_kinds = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid', 'tel_date'])['tel_date'].count().unstack('tel_date').add_prefix('tel_in_date_')
    tel_out_date_all_kinds = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid', 'tel_date'])['tel_date'].count().unstack('tel_date').add_prefix('tel_out_date_')
    tel_date_all_kinds = pd.concat([tel_date_all_kinds, tel_in_date_all_kinds, tel_out_date_all_kinds], axis = 1)

    tel_date_feature = pd.concat([tel_date_static, tel_date_cnt_category, tel_date_all_kinds], axis = 1)
    return tel_date_feature


def get_tel_opp_head_catagory(voice_data):
    tel_opp_head_catagory = voice_data.groupby(['uid', 'tel_opp_head'])['tel_opp_head'].count().unstack('tel_opp_head').add_prefix('tel_opp_head_cat_')
    tel_out_opp_head_catagory = voice_data[voice_data['tel_in_and_out'] == 0].groupby(
        ['uid', 'tel_opp_head'])['tel_opp_head'].count().unstack('tel_opp_head').add_prefix('tel_out_opp_head_cat_')
    tel_in_opp_head_catagory = voice_data[voice_data['tel_in_and_out'] == 1].groupby(
        ['uid', 'tel_opp_head'])['tel_opp_head'].count().unstack('tel_opp_head').add_prefix('tel_in_opp_head_cat_')
        #这里可以做一个in_out_diff, 考虑到比较稀疏 有点复杂， 先放着。
    tel_opp_head_catagory = pd.concat([tel_opp_head_catagory, tel_out_opp_head_catagory], axis = 1)    

    return tel_opp_head_catagory


def get_tel_opp_num_feature(voice_data):
    short_num_kind = voice_data[(voice_data['tel_opp_len'] <= 5)].groupby(['uid', 'tel_opp_num'])['tel_opp_num'].count().unstack('tel_opp_num').add_prefix('tel_opp_num_')
    #long kind试过， 单表分数下降
    #long_num_kind = voice_data[(voice_data['tel_opp_len'] >= 17)].groupby(['uid', 'tel_opp_num'])['tel_opp_num'].count().unstack('tel_opp_num')

#top-K进行onehot
    msg_opp_num_cnt = pd.DataFrame(voice_data['tel_opp_num'].value_counts())
    #调节这里改变模型， 未细致尝试区间。
    hot_tel_num = list(msg_opp_num_cnt[msg_opp_num_cnt['tel_opp_num'] > 800].index)
    hot_tel_opp_num_cnt = []
    for  name in hot_tel_num:
        hot_tel_opp_num_cnt.append(voice_data[voice_data['tel_opp_num'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('hot_tel_opp_num_' + name + '_cnt_'))

    hot_tel_opp_num_cnt.append(voice_data[voice_data['tel_opp_num'].apply(
        lambda x: x not in hot_tel_num)].groupby(['uid'])['uid'].agg(['count']).add_prefix('other_tel_opp_num_' + '_cnt_'))

    hot_tel_opp_num_cnt = pd.concat(hot_tel_opp_num_cnt, axis = 1)
    hot_tel_opp_num_cnt.fillna(0, inplace=True)

    tel_opp_num_feature = pd.concat([short_num_kind, hot_tel_opp_num_cnt], axis = 1)
    return tel_opp_num_feature


def get_tel_type_catagory(voice_data):
    voice_data['tel_call_type'] = voice_data['tel_call_type'].astype('category')

    tel_type_category = voice_data.groupby(['uid', 'tel_call_type'])
    tel_type_category = tel_type_category['tel_call_type'].count().unstack('tel_call_type')
    #tel_type_category.fillna(0, inplace=True)
    #tel_type_category = tel_type_category.apply(lambda x: x.astype(int))
    tel_type_category.columns = list(map(lambda x: 'tel_type_' + str(x) + '_fre', tel_type_category.columns))

    tel_in_type_category = voice_data[voice_data['tel_in_and_out'] == 1].groupby(
        ['uid', 'tel_call_type'])['tel_call_type'].count().unstack('tel_call_type')
    tel_in_type_category.columns = list(map(lambda x: 'tel_type_' + str(x) + '_fre_in', tel_in_type_category.columns))

    tel_out_type_category = voice_data[voice_data['tel_in_and_out'] == 0].groupby(
        ['uid', 'tel_call_type'])['tel_call_type'].count().unstack('tel_call_type')
    tel_out_type_category.columns = list(map(lambda x: 'tel_type_' + str(x) + '_fre_out', tel_out_type_category.columns))

    tel_type_category = pd.concat([tel_type_category, tel_in_type_category, tel_out_type_category], axis = 1)

    tel_type_category.fillna(0, inplace=True)
    tel_type_category = tel_type_category.apply(lambda x: x.astype(int))
    #tel_type_category.head()
    return tel_type_category


def get_tel_len_mean(voice_data):
    #通话平均时长
    tel_out_mean = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid'])['tel_len'].mean()
    tel_in_mean = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid'])['tel_len'].mean()
    tel_len_mean = pd.concat([tel_in_mean, tel_out_mean], join='outer', axis=1)
    tel_len_mean.fillna(0, inplace=True)
    tel_len_mean.columns= ['tel_in_mean', 'tel_out_mean']
    tel_len_mean['in_out_mean_diff'] = tel_len_mean['tel_in_mean'] - tel_len_mean['tel_out_mean']
    #tel_len_mean.head()
    return tel_len_mean


def get_tel_len_sum(voice_data):
    #通话总量
    tel_out_sum = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid'])['tel_len'].sum()
    tel_in_sum = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid'])['tel_len'].sum()
    tel_len_sum = pd.concat([tel_in_sum, tel_out_sum], join='outer', axis=1)
    tel_len_sum.fillna(0, inplace=True)
    tel_len_sum = tel_len_sum.apply(lambda x: x.astype(int))
    tel_len_sum.columns= ['tel_in_sum', 'tel_out_sum']
    tel_len_sum['in_out_sum_diff'] = tel_len_sum['tel_in_sum'] - tel_len_sum['tel_out_sum']
    #tel_len_sum.head()
    return tel_len_sum

def get_tel_len_max(voice_data):
    #通话最长时间
    tel_out_max = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid'])['tel_len'].max()
    tel_in_max = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid'])['tel_len'].max()
    tel_len_max = pd.concat([tel_in_max, tel_out_max], join='outer', axis=1)
    tel_len_max.fillna(0, inplace=True)
    tel_len_max = tel_len_max.apply(lambda x: x.astype(int))
    tel_len_max.columns= ['tel_in_max', 'tel_out_max']
    tel_len_max['in_out_max_diff'] = tel_len_max['tel_in_max'] - tel_len_max['tel_out_max']
    #tel_len_max.head()
    return tel_len_max


def get_tel_len_div(voice_data):
    tel_short_fre = voice_data[voice_data['tel_len'] <= 120].groupby(['uid'])['uid'].count()#少于2min次数
    tel_mid_fre = voice_data[(voice_data['tel_len'] > 120) & (voice_data['tel_len'] <= 600)].groupby(['uid'])['uid'].count() #2-10min 次数
    tel_long_fre = voice_data[voice_data['tel_len'] > 600].groupby(['uid'])['uid'].count() #大于10min次数

    tel_len_div = pd.concat([tel_short_fre,tel_mid_fre,tel_long_fre], axis= 1)
    tel_len_div.columns = ['tel_short_fre','tel_mid_fre','tel_long_fre']
    tel_len_div.fillna(0, inplace=True)
    tel_len_div = tel_len_div.apply(lambda x: x.astype(int))
    #tel_len_div.head()
    return tel_len_div

def get_tel_opp_len_div(voice_data):
    tel_opp_less_5 = voice_data[voice_data['tel_opp_len'] <= 5].groupby(['uid'])['uid'].count()
    tel_opp_less_11 = voice_data[(voice_data['tel_opp_len'] < 11) & (voice_data['tel_opp_len'] > 5)].groupby(['uid'])['uid'].count()
    tel_opp_equal_11 = voice_data[voice_data['tel_opp_len'] == 11].groupby(['uid'])['uid'].count()
    tel_opp_over_11 = voice_data[voice_data['tel_opp_len'] > 11].groupby(['uid'])['uid'].count()

    tel_opp_len_div = pd.concat([tel_opp_less_5,tel_opp_less_11,tel_opp_equal_11,tel_opp_over_11], axis = 1)
    tel_opp_len_div.columns = ['tel_opp_less_5','tel_opp_less_11','tel_opp_equal_11','tel_opp_over_11']
    tel_opp_len_div.fillna(0, inplace=True)
    tel_opp_len_div = tel_opp_len_div.apply(lambda x: x.astype(int))
    #tel_opp_len_div.head()
    return tel_opp_len_div

def get_tel_opp_len_catagory(voice_data):
    tel_opp_len_all_kinds =voice_data.groupby(['uid','tel_opp_len'])['uid'].count().unstack().add_prefix('tel_opp_len_')
    tel_out_opp_len_all_kinds = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid','tel_opp_len'])['uid'].count().unstack().add_prefix('tel_out_opp_len_')
    tel_in_opp_len_all_kinds = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid','tel_opp_len'])['uid'].count().unstack().add_prefix('tel_in_opp_len_')
    #可以做一个in_out_diff,还没写。
    tel_opp_len_all_kinds = pd.concat([tel_opp_len_all_kinds, tel_out_opp_len_all_kinds, tel_in_opp_len_all_kinds], axis = 1)
    return tel_opp_len_all_kinds

def get_tel_hour_feature(voice_data):

    #in_and_out 分类别统计?

    tel_hour_div_0 = voice_data[voice_data['tel_hour'] <= 8].groupby(['uid'])['uid'].count()
    tel_hour_div_1 = voice_data[(voice_data['tel_hour'] > 8) & (voice_data['tel_hour'] <= 12)].groupby(['uid'])['uid'].count()
    tel_hour_div_2 = voice_data[(voice_data['tel_hour'] > 12) & (voice_data['tel_hour'] <= 14)].groupby(['uid'])['uid'].count()
    tel_hour_div_3 = voice_data[(voice_data['tel_hour'] > 14) & (voice_data['tel_hour'] <= 17)].groupby(['uid'])['uid'].count()
    tel_hour_div_4 = voice_data[voice_data['tel_hour'] >= 17].groupby(['uid'])['uid'].count()
    tel_hour_div = pd.concat([tel_hour_div_0,tel_hour_div_1, tel_hour_div_2, tel_hour_div_3, tel_hour_div_4], axis= 1)
    tel_hour_div.columns = ['tel_hour_div_0', 'tel_hour_div_1', 'tel_hour_div_2', 'tel_hour_div_3', 'tel_hour_div_4']

    voice_in_data = voice_data[voice_data['tel_in_and_out'] == 1]
    tel_in_hour_div_0 = voice_in_data[voice_in_data['tel_hour'] <= 8].groupby(['uid'])['uid'].count()
    tel_in_hour_div_1 = voice_in_data[(voice_in_data['tel_hour'] > 8) & (voice_in_data['tel_hour'] <= 12)].groupby(['uid'])['uid'].count()
    tel_in_hour_div_2 = voice_in_data[(voice_in_data['tel_hour'] > 12) & (voice_in_data['tel_hour'] <= 14)].groupby(['uid'])['uid'].count()
    tel_in_hour_div_3 = voice_in_data[(voice_in_data['tel_hour'] > 14) & (voice_in_data['tel_hour'] <= 17)].groupby(['uid'])['uid'].count()
    tel_in_hour_div_4 = voice_in_data[voice_in_data['tel_hour'] >= 17].groupby(['uid'])['uid'].count()
    tel_in_hour_div = pd.concat([tel_in_hour_div_0,tel_in_hour_div_1, tel_in_hour_div_2, tel_in_hour_div_3, tel_in_hour_div_4], axis= 1)
    tel_in_hour_div.columns = ['tel_in_hour_div_0','tel_in_hour_div_1', 'tel_in_hour_div_2', 'tel_in_hour_div_3', 'tel_in_hour_div_4']

    voice_out_data = voice_data[voice_data['tel_in_and_out'] == 0]
    tel_out_hour_div_0 = voice_out_data[voice_out_data['tel_hour'] <= 8].groupby(['uid'])['uid'].count()
    tel_out_hour_div_1 = voice_out_data[(voice_out_data['tel_hour'] > 8) & (voice_out_data['tel_hour'] <= 12)].groupby(['uid'])['uid'].count()
    tel_out_hour_div_2 = voice_out_data[(voice_out_data['tel_hour'] > 12) & (voice_out_data['tel_hour'] <= 14)].groupby(['uid'])['uid'].count()
    tel_out_hour_div_3 = voice_out_data[(voice_out_data['tel_hour'] > 14) & (voice_out_data['tel_hour'] <= 17)].groupby(['uid'])['uid'].count()
    tel_out_hour_div_4 = voice_out_data[voice_out_data['tel_hour'] >= 17].groupby(['uid'])['uid'].count()
    tel_out_hour_div = pd.concat([tel_out_hour_div_0,tel_out_hour_div_1, tel_out_hour_div_2, tel_out_hour_div_3, tel_out_hour_div_4], axis= 1)
    tel_out_hour_div.columns = ['tel_out_hour_div_0','tel_out_hour_div_1', 'tel_out_hour_div_2', 'tel_out_hour_div_3', 'tel_out_hour_div_4']

    tel_hour_div = pd.concat([tel_hour_div, tel_in_hour_div, tel_out_hour_div], axis = 1)
    #tel_hour_category = voice_data.groupby(['uid','tel_hour'])['uid'].count().unstack().add_prefix('tel_hour_cat_')
    #tel_hour_feature = pd.concat([tel_hour_category, tel_hour_div], axis = 1)
    return tel_hour_div


def get_tel_other_feature(voice_data):
    voice_opp_num_unique = voice_data.groupby(['uid'])['tel_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_num_')
    voice_opp_head_unique = voice_data.groupby(['uid'])['tel_opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_')
    voice_opp_len_all_kinds =voice_data.groupby(['uid','tel_opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_')
    voice_opp_len_all_kinds.fillna(0, inplace=True)

    tel_len_static = voice_data.groupby(['uid'])['tel_len'].agg(['std','max','min','median','mean','sum']).add_prefix('tel_len_')
    #tel_in_len_static导致分数下降，可能是过拟合。先加了再说。。。
    tel_in_len_static = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid'])['tel_len'].agg(['std','max','min','median','mean','sum']).add_prefix('tel_len_in_')
    tel_out_len_static = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid'])['tel_len'].agg(['std','max','min','median','mean','sum']).add_prefix('tel_len_out_')
    tel_len_static = pd.concat([tel_len_static, tel_in_len_static, tel_out_len_static], axis = 1)

    tel_len_date_static = voice_data.groupby(['uid', 'tel_date'])['tel_len'].sum().groupby(['uid']).agg(['std','max','min','median','mean','sum']).add_prefix('tel_len_date_')  
    
    yidong_head = ['139', '138','137', '159','151', '158', '150', '136', '187', '182']
    liantong_head = ['130','132', '131', '186']
    dianxin_head = ['189', '180', '133', '156', '153']
    yidong_tel_data = voice_data[voice_data['tel_opp_head'].apply(lambda x: x in yidong_head)]
    liantong_tel_data = voice_data[voice_data['tel_opp_head'].apply(lambda x: x in liantong_head)]
    dianxin_tel_data = voice_data[voice_data['tel_opp_head'].apply(lambda x: x in dianxin_head)]
    other_tel_data = voice_data[voice_data['tel_opp_head'].apply(lambda x: x not in (yidong_head + liantong_head + dianxin_head))]

    yidong_tel_static = yidong_tel_data.groupby(['uid'])['tel_len'].agg(['std','sum', 'max', 'mean']).add_prefix('yidong_tel_len_')
    liantong_tel_static = liantong_tel_data.groupby(['uid'])['tel_len'].agg(['std','sum', 'max', 'mean']).add_prefix('liantong_tel_len_')
    dianxin_tel_static = dianxin_tel_data.groupby(['uid'])['tel_len'].agg(['std','sum', 'max', 'mean']).add_prefix('dianxin_tel_len_')
    other_tel_static = other_tel_data.groupby(['uid'])['tel_len'].agg(['std','sum', 'max', 'mean']).add_prefix('other_tel_len_')
    tel_operator_div = pd.concat([yidong_tel_static,liantong_tel_static, dianxin_tel_static, other_tel_static], axis= 1)

    tel_operator_fre = pd.concat([yidong_tel_data.groupby(['uid'])['uid'].count(), liantong_tel_data.groupby(['uid'])['uid'].count(),
          dianxin_tel_data.groupby(['uid'])['uid'].count()], axis = 1)
    tel_operator_fre.columns = ['yidong_fre', 'liantong_fre', 'dianxin_fre']
    


    gather = pd.concat([voice_opp_num_unique, voice_opp_head_unique,voice_opp_len_all_kinds, tel_len_static,
        tel_len_date_static, tel_operator_div , tel_operator_fre],  axis = 1)
    return gather


def get_voice_feature_matrix():
    voice_data = read_data()
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
    #tel.head()
    return tel


def local_train_and_test(tel):
    train_in_tel = label_train.merge(tel, how='left', left_on='uid',right_on = 'uid')
    train_in_tel.fillna(0, inplace=True) #缺失值0 or -1？

    tel_X_train =train_in_tel.drop(['label'],axis=1)
    tel_y_train  = train_in_tel.label


    tel_local_train_X, tel_local_test_X, local_train_Y, local_test_Y = model_selection.train_test_split(tel_X_train, tel_y_train, test_size=0.25, random_state=42)

    local_test_uid = list(tel_local_test_X['uid'])
    tel_local_train_X = tel_local_train_X.drop(['uid'],axis=1)
    tel_local_test_X = tel_local_test_X.drop(['uid'],axis=1)

    tel_model = tel_train_model(tel_local_train_X, local_train_Y)
    tel_prob = list(tel_model.predict(tel_local_test_X))
    tel_prob =list(map(lambda x: [1 - x, x], tel_prob))
    return local_test_Y, tel_prob


def train(tel):
    train_in_tel = label_train.merge(tel, how='left', left_on='uid',right_on = 'uid')
    train_in_tel.fillna(0, inplace=True) #缺失值0 or -1？

    test_in_tel = label_test.merge(tel, how='left', left_on='uid',right_on = 'uid')
    test_in_tel.fillna(0, inplace=True)


    tel_X_train = train_in_tel.drop(['label', 'uid'],axis=1)
    tel_X_test = test_in_tel.drop(['uid'],axis=1)
    tel_y_train  = train_in_tel.label

    tel_model = tel_train_model(tel_X_train, tel_y_train) # 整个数据集
    tel_prob = list(tel_model.predict(tel_X_test))

    tel_prob =list(map(lambda x: [1 - x, x], tel_prob))

    test_uid = list(label_test['uid'])
    tel_res = list(map(lambda x,y: [x, y], test_uid, list(map(lambda x: x[1], tel_prob))))
    return tel_res





