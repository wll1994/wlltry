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

from sklearn import preprocessing  #预处理，特征归一化
nx = preprocessing.normalize(x)
print nx
print y

#使用卡方检验来对特征进行测试
from sklearn.feature_selection import SelectKBest,chi2
x_new=SelectKBest(chi2,k=500).fit_transform(nx,y)
print x_new
print x_new.shape

#SVM调参数
from sklearn.model_selection import GridSearchCV
parameters = { 'C': [1,5,10,50,100], 'gamma': [0.001,0.01, 0.1, 1, 10 ,100,1000]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new, y)
print('The parameters of the best model are: ')
print(clf.best_params_)

# 用RFE实现特征选择
from sklearn.feature_selection import RFE   #特征选择
# X为样本集合，每个样本为一个数组，数组元素为各个特征值,Y样本的评分
svc=SVC(kernel='linear', C=1,gamma=0.001)
rfe=RFE(estimator=svc,n_features_to_select=20,step=1)
x_new2= rfe.fit_transform(x_new,y)
list2=rfe.get_support(indices=True)
print list2
print x_new2


#SVM调参数
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1,5,10,50,100], 'gamma': [0.001,0.01, 0.1, 1, 10 ,100,1000]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new2, y)
print('The parameters of the best model are: ')
print(clf.best_params_)



#用SVM测试，得到分类正确率
svc = SVC(kernel='rbf', C=1,gamma=100)
#10折交叉验证
result = cross_val_score(svc, x_new2, y, cv=10)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #平均分数,0.88
print result.std()    #标准偏差估计分数
average=0
#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x_new2, y, train_size=0.75)
    model=svc.fit(x_train, y_train)
    a=svc.score(x_test, y_test)
    print '测试集精度 = ', a
    average=average+a
print '平均测试集精度 = ',average/10

print("done in %0.3fs" % (time.time() - t0))
# done in 22.820s
# 不加chi2,时间是done in 1625.203s