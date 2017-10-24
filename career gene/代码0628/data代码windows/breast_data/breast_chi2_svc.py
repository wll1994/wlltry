# -*- coding:utf-8 -*-
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data = np.loadtxt('breast.txt')   #读入txt文件，存为numpy数组，一直报错

np.set_printoptions(threshold=10000)  #设置打印数量的阈值
np.set_printoptions(linewidth=20000)   #使一行能放下一组数据

first=pd.read_csv('breast1.txt',header=None,sep='\t')
#print first.shape
#print first.head()

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
x_new=SelectKBest(chi2,k=60).fit_transform(nx,y)
print x_new
print x_new.shape

#SVM调参数
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1,5,10,50,100], 'gamma': [0.001,0.01, 0.1, 1, 10 ,100,1000]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new, y)
#cv_result = pd.DataFrame.from_dict(clf.cv_results_)
print('The parameters of the best model are: ')
print(clf.best_params_)

#用cross_val_score实现10折验证，进行SVM测试，得到分类正确率
svc = SVC(kernel='rbf', C=10,gamma=100)
result = cross_val_score(svc, x_new, y, cv=10)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #无chi2,平均分数为0.9
print result.std()    #无chi2,标准偏差估计分数为0.16996731712

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
#x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75,random_state=1)
#model=svc.fit(x_train, y_train)
#print '训练集精度 = ', svc.score(x_train, y_train)
#print '测试集精度 = ', svc.score(x_test, y_test)

print("done in %0.3fs" % (time.time() - t0))