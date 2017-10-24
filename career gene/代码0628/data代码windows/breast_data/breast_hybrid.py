# -*- coding:utf-8 -*-
import time
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split

import scipy.io as sio
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
SKB=SelectKBest(chi2,k=200)
x_new=SKB.fit_transform(nx,y)
list=SKB.get_support(indices=True)
print list
#print x_new
#print x_new.shape

from sklearn.feature_selection import SelectKBest,chi2
SKB=SelectKBest(chi2,k=30)
x_new4=SKB.fit_transform(x_new,y)
list4=SKB.get_support(indices=True)
print list4
#print x_new4
#print x_new4.shape

#RFE特征选择
from sklearn.feature_selection import RFE
# X为样本集合，每个样本为一个数组，数组元素为各个特征值,Y样本的评分
svc=SVC(kernel='linear',C=5)
rfe=RFE(estimator=svc,n_features_to_select=30,step=1)
x_new2=rfe.fit_transform(x_new,y)
list2=rfe.get_support(indices=True)
print list2
#print x_new2
#print x_new2.shape

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
#GBDT作为基模型的特征选择
gbdt=SelectFromModel(GradientBoostingClassifier())
x_new3=gbdt.fit_transform(x_new, y)
list3=gbdt.get_support(indices=True)
print list3
#print x_new3.shape
#print x_new3


#基于随机决策树模型的特征选择方法
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf=ExtraTreesClassifier(n_estimators=10, criterion='gini')
clf=clf.fit(x_new,y)
fi=clf.feature_importances_    #输出每个特征的重要性
list=[]
list2=[]
for i in range(200):
    if fi[i]>0.01:
        list.append(i)
        list2.append(fi[i])
print list    #输出特征的序列号
print list2     #输出特征的重要性
model=SelectFromModel(clf,prefit=True)
x_new6=model.transform(x_new)
print x_new6    #输出特征选择后数据集
print x_new6.shape


x_new5=x_new[:,[17,38,50,51,77,110,118,120,179,163,164,182]]
#x_new5=x_new[:,[1,2,3,4]]
#print x_new5

#SVM调参数
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1,5,10,50,100], 'gamma': [0.001,0.01, 0.1, 1, 10 ,100,1000]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new5, y)
print('The parameters of the best model are: ')
print(clf.best_params_)

#用cross_val_score实现10折验证，进行SVM测试，得到分类正确率
svc = SVC(kernel='rbf', C=1,gamma=100)
result = cross_val_score(svc, x_new5, y, cv=10)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #无chi2,平均分数为0.9
print result.std()    #无chi2,标准偏差估计分数为0.16996731712

average=0
average1=0
#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x_new5, y, train_size=0.9,random_state=1)
    model=svc.fit(x_train, y_train)
    a=svc.score(x_test, y_test)
    b=svc.score(x_train, y_train)
    #print '训练集精度 = ',b
    #print '测试集精度 = ', a
    average=average+a
    average1 = average1 + b
print '平均训练集精度 = ',average1/10
print '平均测试集精度 = ',average/10
#print '训练集精度 = ', svc.score(x_train, y_train)
#print '测试集精度 = ', svc.score(x_test, y_test)
#返回给定测试数据和标签的平均精度,无chi2,训练集精度 =  0.891304347826,测试集精度 =  0.9375,
#print '训练集准确率：', accuracy_score(y, svc.predict(nx))  也可以实现

print("done in %0.3fs" % (time.time() - t0))