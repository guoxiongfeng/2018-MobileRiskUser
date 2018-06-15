from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
#Note: 每一轮的Test是不同的uid！ 记得修改这里！
label_test = pd.DataFrame(['u' + str(x + 5000) for x in range(2000)], columns=['uid'])
label_train = pd.read_csv("../data/train/uid_train.txt",header=None,names=['uid','label'], delimiter= '\t')

def feature_matrix_to_train_matrix(features):
    train = label_train.merge(features, how='left', left_on='uid',right_on = 'uid')
    train.fillna(0, inplace=True) 

    test = label_test.merge(features, how='left', left_on='uid',right_on = 'uid')
    test.fillna(0, inplace=True)


    X_train = train.drop(['label', 'uid'],axis=1)
    X_test = test.drop(['uid'],axis=1)
    y_train  = train.label
    return X_train, X_test, y_train
