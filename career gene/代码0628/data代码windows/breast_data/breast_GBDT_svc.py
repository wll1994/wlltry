# -*- coding:utf-8 -*-
#GBDT加SVM，10折验证+2折测试集准确率,选择64个，0.647和0.875.
import time
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

np.set_printoptions(threshold=10000)  #设置打印数量的阈值
np.set_printoptions(linewidth=20000)   #使一行能放下一组数据
np.set_printoptions(suppress=True)     #取消科学计数法为正常输出

first=pd.read_csv('breast1.txt',header=None,sep='\t')
data2=first.T
print data2.shape
print data2.head()

y=data2[0]
x=first[1:].T
print y
print x.shape
print type(x)  #<class 'pandas.core.frame.DataFrame'>
print x.head()
t0 = time.time()

from sklearn import preprocessing
nx = preprocessing.normalize(x)   #预处理，特征归一化

#使用卡方检验来对特征进行测试
from sklearn.feature_selection import SelectKBest,chi2
x_new=SelectKBest(chi2,k=2000).fit_transform(nx,y)
print x_new
print x_new.shape

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
#GBDT作为基模型的特征选择
x_new2=SelectFromModel(GradientBoostingClassifier()).fit_transform(x_new, y)
print x_new2.shape


#用cross_val_score实现10折验证，进行SVM测试，得到分类正确率
svc = SVC(kernel='linear', C=20)
result = cross_val_score(svc, x_new2, y, cv=8)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #无chi2,平均分数为0.9
print result.std()    #无chi2,标准偏差估计分数为0.16996731712

print("done in %0.3fs" % (time.time() - t0))

average=0
#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x_new2, y, train_size=0.75,random_state=0)
    model=svc.fit(x_train, y_train)
    a=svc.score(x_test, y_test)
    print '测试集精度 = ', a
    average=average+a
print '平均测试集精度 = ',average/10

#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
#x_train, x_test, y_train, y_test = train_test_split(x_new2, y, train_size=0.6,random_state=1)
#model=svc.fit(x_train, y_train)
#print '训练集精度 = ', svc.score(x_train, y_train)
#print '测试集精度 = ', svc.score(x_test, y_test)
#返回给定测试数据和标签的平均精度,无chi2,训练集精度 =  0.891304347826,测试集精度 =  0.9375,
#print '训练集准确率：', accuracy_score(y, svc.predict(nx))  也可以实现

print("done in %0.3fs" % (time.time() - t0))