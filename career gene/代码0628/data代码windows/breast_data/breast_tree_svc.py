# -*- coding:utf-8 -*-
import time
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=10000)  #设置打印数量的阈值
np.set_printoptions(linewidth=20000)   #使一行能放下一组数据

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

from sklearn import preprocessing  #预处理，特征归一化
nx = preprocessing.normalize(x)
print nx

#基于随机决策树模型的特征选择方法
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf=ExtraTreesClassifier(n_estimators=10, criterion='gini')
clf=clf.fit(nx,y)
fi=clf.feature_importances_    #输出每个特征的重要性
list=[]
list2=[]
for i in range(22283):
    if fi[i]>0:
        list.append(i)
        list2.append(fi[i])
print list    #输出特征的序列号
print list2     #输出特征的重要性

model=SelectFromModel(clf,prefit=True)
x_new=model.transform(nx)
print x_new    #输出特征选择后数据集
print x_new.shape
x_new2=x_new[:,:20]   #选择前20个基因
print x_new2
print x_new2.shape

#用cross_val_score实现10折验证，进行SVM测试，得到分类正确率
svc = SVC(kernel='linear', C=5)
result = cross_val_score(svc, x_new, y, cv=10)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #无chi2,平均分数为0.9
print result.std()    #无chi2,标准偏差估计分数为0.16996731712

#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75,random_state=1)
model=svc.fit(x_train, y_train)
print '训练集精度 = ', svc.score(x_train, y_train)
print '测试集精度 = ', svc.score(x_test, y_test)
#返回给定测试数据和标签的平均精度,无chi2,训练集精度 =  0.891304347826,测试集精度 =  0.9375,
#print '训练集准确率：', accuracy_score(y, svc.predict(nx))  也可以实现

print("done in %0.3fs" % (time.time() - t0))