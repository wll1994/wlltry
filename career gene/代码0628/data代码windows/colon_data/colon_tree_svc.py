# -*- coding:utf-8 -*-
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=10000)  #设置打印数量的阈值
np.set_printoptions(linewidth=20000)   #使一行能放下一组数据

data=sio.loadmat('colon_tumor.mat')   #读取MATLAB数据,data为字典对象
#print data                           #输出读取的数据，发现有6个键值对，第一个target和第四个ColonDataSet是需要的
data_data=data['ColonDataSet']       #此为数组，第一列为分类值后标签，第二列到最后为特征值
np.set_printoptions(linewidth=2000)   #使一行能放下一组数据
print data_data
print data_data.shape
y=data_data[:,0]  #读取标签
print y
x=data_data[:,1:]   #读取特征值
print x

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
for i in range(2000):
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


average=0
#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75)
    model=svc.fit(x_train, y_train)
    a=svc.score(x_test, y_test)
    print '测试集精度 = ', a
    average=average+a
print '平均测试集精度 = ',average/10

print("done in %0.3fs" % (time.time() - t0))