# -*- coding:utf-8 -*-
#卡方检验加SVM，10折验证+2折测试集准确率,选择29个，0.9和0.9375.
import time
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

data=sio.loadmat('colon_tumor.mat')   #读取MATLAB数据,data为字典对象  输出读取的数据，发现有6个键值对，第一个target和第四个ColonDataSet是需要的
data_data=data['ColonDataSet']       #此为数组，第一列为分类值后标签，第二列到最后为特征值
np.set_printoptions(linewidth=2000)   #使一行能放下一组数据
print data_data
print data_data.shape

y=data_data[:,0]  #读取标签
print y
print y.shape
x=data_data[:,1:]   #读取特征值
print x
print x.shape

#查看标签为0的序号
list3=[0]
for i in range(62):
    if y[i]<=0:
        list3.append(i)
print list3

t0 = time.time()

from sklearn import preprocessing
nx = preprocessing.normalize(x)   #预处理，特征归一化

#使用卡方检验来对特征进行测试
from sklearn.feature_selection import SelectKBest,chi2
x_new=SelectKBest(chi2,k=29).fit_transform(nx,y)
print x_new
print x_new.shape



#SVM调参数
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1,5,10,50,100], 'gamma': [0.001,0.01, 0.1, 1, 10 ,100,1000]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new, y)
print('The parameters of the best model are: ')
print(clf.best_params_)


#用cross_val_score实现10折验证，进行SVM测试，得到分类正确率
svc = SVC(kernel='rbf', C=1,gamma=10 )
result = cross_val_score(svc, x_new, y, cv=10)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #无chi2,平均分数为0.9
print result.std()    #无chi2,标准偏差估计分数为0.16996731712

print("done in %0.3fs" % (time.time() - t0))

average=0
average1=0
#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75)
    model=svc.fit(x_train, y_train)
    a=svc.score(x_test, y_test)
    b=svc.score(x_train, y_train)
    #print '训练集精度 = ',b
    #print '测试集精度 = ', a
    average=average+a
    average1 = average1 + b
print '平均训练集精度 = ',average1/10
print '平均测试集精度 = ',average/10

#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
#x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75,random_state=0)
#model=svc.fit(x_train, y_train)
#print '训练集精度 = ', svc.score(x_train, y_train)
#print '测试集精度 = ', svc.score(x_test, y_test)
#返回给定测试数据和标签的平均精度,无chi2,训练集精度 =  0.891304347826,测试集精度 =  0.9375,
#print '训练集准确率：', accuracy_score(y, svc.predict(nx))  也可以实现

print("done in %0.3fs" % (time.time() - t0))