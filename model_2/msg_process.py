import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBRegressor

#Note: 每一轮的Test是不同的uid！ 记得修改这里！
label_test = pd.DataFrame(['u' + str(x + 7000) for x in range(3000)], columns=['uid'])
label_train = pd.read_csv("../data/train/uid_train.txt",header=None,names=['uid','label'], delimiter= '\t')


#读入数据，并得到一些初步信息
def read_data():
    names_smg = ['uid','smg_opp_num','smg_opp_head','smg_opp_len','smg_start_time','smg_in_and_out']
    smg_train = pd.read_csv("../data/train/sms_train.txt",header=None,encoding='utf-8',names = names_smg,index_col = False, delimiter='\t',
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
    smg_data = pd.concat([smg_train, smg_test])
    smg_data['smg_hour'] = smg_data['smg_start_time'].apply(lambda x: int(x[2:4]))
    smg_data['smg_date'] = smg_data['smg_start_time'].apply(lambda x: int(x[0:2]))
    return smg_data


def get_msg_date_feature(msg_data):

    #分 in out 计算用户按日统计的一些统计量
    msg_date_static = msg_data.groupby(['uid', 'smg_date'])['uid'].count().groupby(['uid']).agg(
        ['mean', 'count', 'sum', 'std']).add_prefix('msg_date_')
    msg_in_date_static = msg_data[msg_data['smg_in_and_out'] == 1].groupby(['uid', 'smg_date'])['uid'].count().groupby(['uid']).agg(
        ['mean', 'count', 'sum', 'std']).add_prefix('msg_in_date_')
    msg_out_date_static = msg_data[msg_data['smg_in_and_out'] == 0].groupby(['uid', 'smg_date'])['uid'].count().groupby(['uid']).agg(
        ['mean', 'count', 'sum', 'std']).add_prefix('msg_out_date_')
    msg_date_static = pd.concat([msg_date_static, msg_in_date_static, msg_out_date_static], axis = 1)

    #不能按照下面这种写法， 处理空数据一定的问题
    #msg_in_out_diff = msg_out_date_static.copy(deep=True)
    #msg_in_out_diff.columns = msg_in_date_static.columns
    #msg_in_out_diff = msg_in_date_static - msg_in_out_diff
    #msg_date_static = pd.concat([msg_date_static, msg_in_date_static, msg_out_date_static, msg_in_out_diff], axis = 1)

    #计算该类特征的in out 相差值
    msg_date_static.fillna(0, inplace=True)
    msg_date_static['msg_date_in_out_diff_mean'] = msg_date_static['msg_in_date_mean'] - msg_date_static['msg_out_date_mean']
    msg_date_static['msg_date_in_out_diff_count'] = msg_date_static['msg_in_date_count'] - msg_date_static['msg_out_date_count']
    msg_date_static['msg_date_in_out_diff_sum'] = msg_date_static['msg_in_date_sum'] - msg_date_static['msg_out_date_sum']
    msg_date_static['msg_date_in_out_diff_std'] = msg_date_static['msg_in_date_std'] - msg_date_static['msg_out_date_std']


    #按照5天为一个日期的bag， 统计一共45/ 5 = 9 个bag的统计量（这里统计量即为次数）
    date_bag_size = 5
    msg_date_cnt_category = []
    for cat_num in range(45 // date_bag_size):
        msg_date_cnt_category.append((msg_data[(msg_data['smg_date'] - 1) // date_bag_size == cat_num].groupby(
            ['uid'])['uid'].agg(['count'])).add_prefix('msg_date_cnt_cat_' + str(cat_num) + '_') )
    msg_date_cnt_category = pd.concat(msg_date_cnt_category, axis = 1)

    #这部分是不做bag， 直接对45 天每天的数据统计次数
    msg_date_all_kinds = msg_data.groupby(['uid', 'smg_date'])['smg_date'].count().unstack('smg_date').add_prefix('msg_date_')
    msg_in_date_all_kinds = msg_data[msg_data['smg_in_and_out'] == 1].groupby(['uid', 'smg_date'])['smg_date'].count().unstack('smg_date').add_prefix('msg_in_date_')
    msg_out_date_all_kinds = msg_data[msg_data['smg_in_and_out'] == 0].groupby(['uid', 'smg_date'])['smg_date'].count().unstack('smg_date').add_prefix('msg_out_date_')
    msg_date_all_kinds = pd.concat([msg_date_all_kinds, msg_in_date_all_kinds, msg_out_date_all_kinds], axis = 1)
    #这里可以做一个in_out_diff, 考虑到比较稀疏 有点复杂， 先放着。

    msg_date_feature = pd.concat([msg_date_static, msg_date_cnt_category, msg_date_all_kinds], axis = 1)
    return msg_date_feature


def get_msg_opp_head_catagory(msg_data):
    #把head信息做one-hot处理
    #这里也是分别分了 in out 来做
    msg_opp_head_catagory = msg_data.groupby(['uid', 'smg_opp_head'])['smg_opp_head'].count().unstack('smg_opp_head').add_prefix('smg_opp_head_')
    msg_out_opp_head_catagory = msg_data[msg_data['smg_in_and_out'] == 0].groupby(
        ['uid', 'smg_opp_head'])['smg_opp_head'].count().unstack('smg_opp_head').add_prefix('msg_out_opp_head_cat_')
    msg_in_opp_head_catagory = msg_data[msg_data['smg_in_and_out'] == 1].groupby(
        ['uid', 'smg_opp_head'])['smg_opp_head'].count().unstack('smg_opp_head').add_prefix('msg_in_opp_head_cat_')
    #这里可以做一个in_out_diff, 考虑到比较稀疏 有点复杂， 先放着。
    msg_opp_head_catagory = pd.concat([msg_opp_head_catagory, msg_out_opp_head_catagory, msg_in_opp_head_catagory], axis = 1)
    return msg_opp_head_catagory
    #t.apply(lambda x: x.astype(int))
    #t.count()


#利用msg_head 区分运营商
def get_msg_operator_div(msg_data):
    #不同运营商head的大致列表
    yidong_head = ['139', '138','137', '159','151', '158', '150', '136', '187', '182']
    liantong_head = ['130','132', '131', '186']
    dianxin_head = ['189', '180', '133', '156', '153']
    #查资料可知， 000开头为国际电话
    foreign_head = ['000']
    yidong_msg_data = msg_data[msg_data['smg_opp_head'].apply(lambda x: x in yidong_head)]
    liantong_msg_data = msg_data[msg_data['smg_opp_head'].apply(lambda x: x in liantong_head)]
    dianxin_msg_data = msg_data[msg_data['smg_opp_head'].apply(lambda x: x in dianxin_head)]
    other_msg_data = msg_data[msg_data['smg_opp_head'].apply(lambda x: x not in (yidong_head + liantong_head + dianxin_head + foreign_head))]

    #下面一大段代码， 就是在统计区分不同运营商的短信次数， 并且也是分in out， 最后计算in out的差值

    yidong_smg_cnt = yidong_msg_data.groupby(['uid'])['uid'].agg(['count']).add_prefix('yidong_msg_')
    liantong_smg_cnt = liantong_msg_data.groupby(['uid'])['uid'].agg(['count']).add_prefix('liantong_msg_')
    dianxin_smg_cnt = dianxin_msg_data.groupby(['uid'])['uid'].agg(['count']).add_prefix('dianxin_msg_')
    other_smg_cnt = other_msg_data.groupby(['uid'])['uid'].agg(['count']).add_prefix('other_msg_')
    smg_operator_div = pd.concat([yidong_smg_cnt, liantong_smg_cnt, dianxin_smg_cnt, other_smg_cnt], axis= 1)
    #in
    yidong_smg_in_cnt = yidong_msg_data[yidong_msg_data['smg_in_and_out'] == 1].groupby(['uid'])['uid'].agg(['count']).add_prefix('yidong_msg_in_')
    liantong_smg_in_cnt = liantong_msg_data[liantong_msg_data['smg_in_and_out'] == 1].groupby(['uid'])['uid'].agg(['count']).add_prefix('liantong_msg_in_')
    dianxin_smg_in_cnt = dianxin_msg_data[dianxin_msg_data['smg_in_and_out'] == 1].groupby(['uid'])['uid'].agg(['count']).add_prefix('dianxin_msg_in_')
    other_smg_in_cnt = other_msg_data[other_msg_data['smg_in_and_out'] == 1].groupby(['uid'])['uid'].agg(['count']).add_prefix('other_msg_in_')
    smg_in_operator_div = pd.concat([yidong_smg_in_cnt, liantong_smg_in_cnt, dianxin_smg_in_cnt, other_smg_in_cnt], axis= 1)
    #out
    yidong_smg_out_cnt = yidong_msg_data[yidong_msg_data['smg_in_and_out'] == 0].groupby(['uid'])['uid'].agg(['count']).add_prefix('yidong_msg_out_')
    liantong_smg_out_cnt = liantong_msg_data[liantong_msg_data['smg_in_and_out'] == 0].groupby(['uid'])['uid'].agg(['count']).add_prefix('liantong_msg_out_')
    dianxin_smg_out_cnt = dianxin_msg_data[dianxin_msg_data['smg_in_and_out'] == 0].groupby(['uid'])['uid'].agg(['count']).add_prefix('dianxin_msg_out_')
    other_smg_out_cnt = other_msg_data[other_msg_data['smg_in_and_out'] == 0].groupby(['uid'])['uid'].agg(['count']).add_prefix('other_msg_out_')
    smg_out_operator_div = pd.concat([yidong_smg_out_cnt, liantong_smg_out_cnt, dianxin_smg_out_cnt, other_smg_out_cnt], axis= 1)

    #diff
    yidong_smg_diff_cnt = yidong_smg_in_cnt['yidong_msg_in_count'] - yidong_smg_out_cnt['yidong_msg_out_count']
    liantong_smg_diff_cnt = liantong_smg_in_cnt['liantong_msg_in_count'] - liantong_smg_out_cnt['liantong_msg_out_count']
    dianxin_smg_diff_cnt = dianxin_smg_in_cnt['dianxin_msg_in_count'] - dianxin_smg_out_cnt['dianxin_msg_out_count']
    other_smg_diff_cnt = other_smg_in_cnt['other_msg_in_count'] - other_smg_out_cnt['other_msg_out_count']
    
    smg_operator_diff = pd.concat([yidong_smg_diff_cnt, liantong_smg_diff_cnt, dianxin_smg_diff_cnt, other_smg_diff_cnt], axis= 1)
    smg_operator_diff.columns = ['yidong_smg_diff_cnt', 'liantong_smg_diff_cnt', 'dianxin_smg_diff_cnt', 'other_smg_diff_cnt']
    smg_operator_div = pd.concat([smg_operator_div, smg_in_operator_div, smg_out_operator_div, smg_operator_diff], axis = 1)

    return smg_operator_div


def get_msg_opp_num_feature(msg_data):
    #对于msg_opp_num 由于取值过多， 因此选取top-K做one-hot
    msg_opp_num_cnt = pd.DataFrame(msg_data['smg_opp_num'].value_counts())
    #调节这里改变模型 注：此范围区间已经经过调整，基本是最佳的。
    #在3000前的有几个值特别大， 怀疑是某个公用号码
    hot_msg_num = list(msg_opp_num_cnt[(msg_opp_num_cnt['smg_opp_num'] < 3000) & (msg_opp_num_cnt['smg_opp_num'] > 1000)].index)

    hot_msg_opp_num_cnt = []
    for  name in hot_msg_num:
        hot_msg_opp_num_cnt.append(msg_data[msg_data['smg_opp_num'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('hot_msg_opp_num_' + name + '_cnt_'))
        
    hot_msg_opp_num_cnt.append(msg_data[msg_data['smg_opp_num'].apply(
        lambda x: x not in hot_msg_num)].groupby(['uid'])['uid'].agg(['count']).add_prefix('other_msg_opp_num_' + '_cnt_'))
    hot_msg_opp_num_cnt = pd.concat(hot_msg_opp_num_cnt, axis = 1)
    hot_msg_opp_num_cnt.fillna(0, inplace=True)
    return hot_msg_opp_num_cnt


def get_msg_short_num_kind(smg_data) :
    #smg_opp_num 处理
    #先过滤  再one-hot处理 不然会使特征维数过多
    short_num_kind = smg_data[smg_data['smg_opp_len']  <= 8].groupby(['uid', 'smg_opp_num'])['smg_opp_num'].count().unstack('smg_opp_num')
    short_num_kind.fillna(0, inplace=True)
    short_num_kind = short_num_kind.apply(lambda x: x.astype(int))
    #short_num_kind.head()
    return short_num_kind

def get_msg_opp_len_div(smg_data):
    #对于不同号码长度， 做若干个分类， 以5， 11为分界点。
    #计算对于每个类别的频数，以及每个类别占总量的比重
    smg_opp_cnt = smg_data.groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_')
    smg_opp_less_5_cnt = smg_data[smg_data['smg_opp_len'] <= 5].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_less_5_')
    smg_opp_less_11_cnt = smg_data[(smg_data['smg_opp_len'] < 11) & (smg_data['smg_opp_len'] > 5)].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_less_11_')
    smg_opp_equal_11_cnt = smg_data[smg_data['smg_opp_len'] == 11].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_equal_11_')
    smg_opp_over_11_cnt = smg_data[smg_data['smg_opp_len'] > 11].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_over_11_')
    smg_opp_len_div_cnt = pd.concat([smg_opp_cnt, smg_opp_less_5_cnt,smg_opp_less_11_cnt,smg_opp_equal_11_cnt,smg_opp_over_11_cnt], axis = 1)
    #smg_opp_len_div_cnt.columns = ['smg_opp_less_5','smg_opp_less_11','smg_opp_equal_11','smg_opp_over_11']
    smg_opp_len_div_cnt.fillna(0, inplace=True)

    smg_opp_less_5_ratio = pd.concat([smg_opp_less_5_cnt['smg_opp_less_5_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_less_5_cnt['smg_opp_less_5_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_less_11_ratio = pd.concat([smg_opp_less_11_cnt['smg_opp_less_11_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_less_11_cnt['smg_opp_less_11_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_equal_11_ratio = pd.concat([smg_opp_equal_11_cnt['smg_opp_equal_11_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_equal_11_cnt['smg_opp_equal_11_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_over_11_ratio = pd.concat([smg_opp_over_11_cnt['smg_opp_over_11_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_over_11_cnt['smg_opp_over_11_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_len_ratio = pd.concat([smg_opp_less_5_ratio, smg_opp_less_11_ratio, smg_opp_equal_11_ratio, smg_opp_over_11_ratio], axis= 1)
    smg_opp_len_ratio.columns = ['smg_opp_less_5_unique_ratio', 'smg_opp_less_5_cnt_ratio', 'smg_opp_less_11_unique_ratio', 'smg_opp_less_11_cnt_ratio',
    'smg_opp_equal_11_unique_ratio', 'smg_opp_equal_11_cnt_ratio', 'smg_opp_over_11_unique_ratio', 'smg_opp_over_11_cnt_ratio']

    smg_opp_len_div = pd.concat([smg_opp_len_div_cnt ,smg_opp_len_ratio], axis = 1)
    #smg_opp_len_div.head()
    return smg_opp_len_div

def get_msg_in_and_out(smg_data):
    #in-and-out 的频数统计与差值
    smg_in_and_out = smg_data.groupby(['uid', 'smg_in_and_out'])['smg_in_and_out'].count().unstack('smg_in_and_out')
    smg_in_and_out.columns=['smg_in_cnt','smg_out_cnt']
    smg_in_and_out.fillna(0, inplace=True)
    smg_in_and_out['smg_in_out_cnt_diff'] = smg_in_and_out['smg_in_cnt'] - smg_in_and_out['smg_out_cnt']
    #smg_in_and_out.head()
    return smg_in_and_out

#对opp_head进行value_count()可知000开头的境外号码占总数最多，故特别安排一个特征
def get_smg_foreign_fre(smg_data):
    foreign_fre = pd.DataFrame(smg_data[smg_data['smg_opp_head'] == '000'].groupby(['uid'])['uid'].count())
    foreign_fre.columns = ['foreign_fre']
    #foreign_fre.head()
    return foreign_fre




#smg_hour 还不知道怎么使用，先不放了
#未使用 msg hour 加上之后正确率反而下降
#以下是msg_hour的处理代码
def get_msg_hour_feature(msg_data):
    #直接one-hot 会导致下降一点， 考虑分类试试
    #按照人们日常作息进行分类，并且分 in out 统计
    msg_hour_div_0 = msg_data[msg_data['smg_hour'] <= 8].groupby(['uid'])['uid'].count()
    msg_hour_div_1 = msg_data[(msg_data['smg_hour'] > 8) & (msg_data['smg_hour'] <= 12)].groupby(['uid'])['uid'].count()
    msg_hour_div_2 = msg_data[(msg_data['smg_hour'] > 12) & (msg_data['smg_hour'] <= 14)].groupby(['uid'])['uid'].count()
    msg_hour_div_3 = msg_data[(msg_data['smg_hour'] > 14) & (msg_data['smg_hour'] <= 17)].groupby(['uid'])['uid'].count()
    msg_hour_div_4 = msg_data[msg_data['smg_hour'] >= 17].groupby(['uid'])['uid'].count()
    msg_hour_div = pd.concat([msg_hour_div_0,msg_hour_div_1, msg_hour_div_2, msg_hour_div_3, msg_hour_div_4], axis= 1)
    msg_hour_div.columns = ['msg_hour_div_0', 'msg_hour_div_1', 'msg_hour_div_2', 'msg_hour_div_3', 'msg_hour_div_4']

    msg_in_data = msg_data[msg_data['smg_in_and_out'] == 1]
    msg_in_hour_div_0 = msg_in_data[msg_in_data['smg_hour'] <= 8].groupby(['uid'])['uid'].count()
    msg_in_hour_div_1 = msg_in_data[(msg_in_data['smg_hour'] > 8) & (msg_in_data['smg_hour'] <= 12)].groupby(['uid'])['uid'].count()
    msg_in_hour_div_2 = msg_in_data[(msg_in_data['smg_hour'] > 12) & (msg_in_data['smg_hour'] <= 14)].groupby(['uid'])['uid'].count()
    msg_in_hour_div_3 = msg_in_data[(msg_in_data['smg_hour'] > 14) & (msg_in_data['smg_hour'] <= 17)].groupby(['uid'])['uid'].count()
    msg_in_hour_div_4 = msg_in_data[msg_in_data['smg_hour'] >= 17].groupby(['uid'])['uid'].count()
    msg_in_hour_div = pd.concat([msg_in_hour_div_0,msg_in_hour_div_1, msg_in_hour_div_2, msg_in_hour_div_3, msg_in_hour_div_4], axis= 1)
    msg_in_hour_div.columns = ['msg_in_hour_div_0','msg_in_hour_div_1', 'msg_in_hour_div_2', 'msg_in_hour_div_3', 'msg_in_hour_div_4']

    msg_out_data = msg_data[msg_data['smg_in_and_out'] == 0]
    msg_out_hour_div_0 = msg_out_data[msg_out_data['smg_hour'] <= 8].groupby(['uid'])['uid'].count()
    msg_out_hour_div_1 = msg_out_data[(msg_out_data['smg_hour'] > 8) & (msg_out_data['smg_hour'] <= 12)].groupby(['uid'])['uid'].count()
    msg_out_hour_div_2 = msg_out_data[(msg_out_data['smg_hour'] > 12) & (msg_out_data['smg_hour'] <= 14)].groupby(['uid'])['uid'].count()
    msg_out_hour_div_3 = msg_out_data[(msg_out_data['smg_hour'] > 14) & (msg_out_data['smg_hour'] <= 17)].groupby(['uid'])['uid'].count()
    msg_out_hour_div_4 = msg_out_data[msg_out_data['smg_hour'] >= 17].groupby(['uid'])['uid'].count()
    msg_out_hour_div = pd.concat([msg_out_hour_div_0,msg_out_hour_div_1, msg_out_hour_div_2, msg_out_hour_div_3, msg_out_hour_div_4], axis= 1)
    msg_out_hour_div.columns = ['msg_out_hour_div_0','msg_out_hour_div_1', 'msg_out_hour_div_2', 'msg_out_hour_div_3', 'msg_out_hour_div_4']

    msg_hour_div = pd.concat([msg_hour_div, msg_in_hour_div, msg_out_hour_div], axis = 1)




    msg_hour_category = msg_data.groupby(['uid','smg_hour'])['uid'].count().unstack().add_prefix('sms_hour_div_')

    #分类结果也要更差
    #msg_hour_div_0 = msg_data[msg_data['smg_hour'] <= 6].groupby(['uid'])['uid'].count()
    #msg_hour_div_1 = msg_data[(msg_data['smg_hour'] > 6) & (msg_data['smg_hour'] <= 12)].groupby(['uid'])['uid'].count()
    #msg_hour_div_2 = msg_data[(msg_data['smg_hour'] > 12) & (msg_data['smg_hour'] <= 18)].groupby(['uid'])['uid'].count()
    #msg_hour_div_3 = msg_data[msg_data['smg_hour'] > 18].groupby(['uid'])['uid'].count()
    #msg_hour_div = pd.concat([msg_hour_div_0, msg_hour_div_1, msg_hour_div_2, msg_hour_div_3], axis= 1)
    #msg_hour_div.columns = ['msg_hour_div_0', 'msg_hour_div_1', 'msg_hour_div_2', 'msg_hour_div_3']
    msg_hour_feature = pd.concat([msg_hour_div, msg_hour_category], axis = 1)
    return msg_hour_feature

#对opp_len 进行one-hot处理
def get_msg_opp_len_catagory(msg_data):
    smg_opp_len_all_kinds =msg_data.groupby(['uid','smg_opp_len'])['uid'].count().unstack().add_prefix('msg_opp_len_')
    smg_out_opp_len_all_kinds = msg_data[msg_data['smg_in_and_out'] == 0].groupby(['uid','smg_opp_len'])['uid'].count().unstack().add_prefix('msg_out_opp_len_')
    smg_in_opp_len_all_kinds = msg_data[msg_data['smg_in_and_out'] == 1].groupby(['uid','smg_opp_len'])['uid'].count().unstack().add_prefix('msg_in_opp_len_')
    #可以做一个in_out_diff,还没写。
    smg_opp_len_all_kinds = pd.concat([smg_opp_len_all_kinds, smg_out_opp_len_all_kinds, smg_in_opp_len_all_kinds], axis = 1)
    return smg_opp_len_all_kinds

#后续提取的特征都放这里， 方便加入删除
def get_smg_other_feature(smg_data):
    #重复了统计不同的opp_num和opp_head
    smg_opp_num_unique = smg_data.groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('smg_opp_num_')
    smg_opp_head_unique = smg_data.groupby(['uid'])['smg_opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('smg_opp_head_')

    #分in_out统计unique的数量
    smg_in_opp_num_unique = smg_data[smg_data['smg_in_and_out'] == 1].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('smg_in_opp_num_')

    smg_in_opp_head_unique = smg_data[smg_data['smg_in_and_out'] == 1].groupby(['uid'])['smg_opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('smg_in_opp_head_')
    
    smg_out_opp_num_unique = smg_data[smg_data['smg_in_and_out'] == 0].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('smg_out_opp_num_')
    
    smg_out_opp_head_unique = smg_data[smg_data['smg_in_and_out'] == 0].groupby(['uid'])['smg_opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('smg_out_opp_head_')
    
    #unique diff
    smg_in_out_opp_num_unique_diff = smg_in_opp_num_unique['smg_in_opp_num_unique_count'] - smg_out_opp_num_unique['smg_out_opp_num_unique_count']
    #smg_in_out_opp_num_unique_diff.columns = ['smg_in_out_opp_num_unique_diff']
    smg_in_out_opp_head_unique_diff = smg_in_opp_head_unique['smg_in_opp_head_unique_count'] - smg_out_opp_head_unique['smg_out_opp_head_unique_count']
    #smg_in_out_opp_head_unique_diff.columns = ['smg_in_out_opp_head_unique_diff']

    smg_opp_unique_static = pd.concat([smg_opp_num_unique, smg_opp_head_unique, smg_in_opp_num_unique, smg_in_opp_head_unique,
        smg_out_opp_head_unique, smg_out_opp_num_unique, smg_in_out_opp_num_unique_diff, smg_in_out_opp_head_unique_diff], axis = 1)
    smg_opp_unique_static.columns = ['smg_opp_num_unique', 'smg_opp_head_unique', 'smg_in_opp_num_unique', 'smg_in_opp_head_unique',
        'smg_out_opp_head_unique', 'smg_out_opp_num_unique', 'smg_in_out_opp_num_unique_diff', 'smg_in_out_opp_head_unique_diff']
    

    #重复
    #smg_opp_len_all_kinds =smg_data.groupby(['uid','smg_opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_')
    #smg_opp_len_all_kinds.fillna(0, inplace=True)

    smg_in_date_static = smg_data[smg_data['smg_in_and_out'] == 1].groupby(
        ['uid', 'smg_date'])['uid'].count().groupby(['uid']).agg(['std','max','median']).add_prefix('smg_in_date_')
    smg_out_date_static = smg_data[smg_data['smg_in_and_out'] == 0].groupby(
        ['uid', 'smg_date'])['uid'].count().groupby(['uid']).agg(['std','max','median']).add_prefix('smg_out_date_')
    smg_date_static = smg_data.groupby(
        ['uid', 'smg_date'])['uid'].count().groupby(['uid']).agg(['std','max','median']).add_prefix('smg_date_')


    short_num_kind = smg_data[(smg_data['smg_opp_len']  <= 7)].groupby(['uid', 'smg_opp_num'])['smg_opp_num'].count().unstack('smg_opp_num')

    smg_date_static = pd.concat([smg_in_date_static, smg_out_date_static, smg_date_static], axis = 1)

    gather = pd.concat([smg_opp_unique_static, short_num_kind, smg_date_static],  axis = 1)
    return gather

#整合msg的特征构建特征矩阵
def get_msg_feature_matrix():
    msg_data = read_data()

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
    return smg


######下面部分是开始时的一个local test，被后续版本的cv所替代了
def local_train_and_test(smg):
    train_in_smg = label_train.merge(smg, how='left', left_on='uid',right_on = 'uid')
    train_in_smg.fillna(0, inplace=True)
    #train_in_smg.head()
    msg_X_train =train_in_smg.drop(['label'],axis=1)
    msg_y_train  = train_in_smg.label

    msg_local_train_X, msg_local_test_X, local_train_Y, local_test_Y = model_selection.train_test_split(msg_X_train, msg_y_train, test_size=0.25, random_state=42)
    local_test_uid = list(msg_local_test_X['uid'])
    msg_local_train_X = msg_local_train_X.drop(['uid'],axis=1)
    msg_local_test_X = msg_local_test_X.drop(['uid'],axis=1)

    msg_model = msg_train_model(msg_local_train_X, local_train_Y)
    msg_prob = msg_model.predict_proba(msg_local_test_X)
    return local_test_Y, msg_prob

#Note: this train is for the final result.(submit version)
#返回值是一个概率列表，按顺序对应测试集A u5000-u6999
def train(smg):
    train_in_smg = label_train.merge(smg, how='left', left_on='uid',right_on = 'uid')
    train_in_smg.fillna(0, inplace=True)
    #train_in_smg.head()

    test_in_smg = label_test.merge(smg, how='left', left_on='uid',right_on = 'uid')
    test_in_smg.fillna(0, inplace=True)
    #test_in_smg.head()
    msg_X_train = train_in_smg.drop(['label', 'uid'],axis=1)
    msg_X_test = test_in_smg.drop(['uid'],axis=1)
    msg_y_train  = train_in_smg.label

    msg_model = msg_train_model(msg_X_train, msg_y_train) # 整个数据集
    msg_prob = msg_model.predict_proba(msg_X_test)

    test_uid = list(label_test['uid'])
    msg_res = list(map(lambda x,y: [x, y], test_uid, list(map(lambda x: x[1], msg_prob))))
    return msg_res

