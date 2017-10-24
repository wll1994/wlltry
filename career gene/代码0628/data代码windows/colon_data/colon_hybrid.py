# -*- coding:utf-8 -*-
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

t0 = time.time()

from sklearn import preprocessing
nx = preprocessing.normalize(x)   #预处理，特征归一化

#使用卡方检验来对特征进行测试
from sklearn.feature_selection import SelectKBest,chi2
SKB=SelectKBest(chi2,k=70)
x_new=SKB.fit_transform(nx,y)
list=SKB.get_support(indices=True)
print list
#print x_new
#print x_new.shape

from sklearn.feature_selection import SelectKBest,chi2
SKB=SelectKBest(chi2,k=15)
x_new4=SKB.fit_transform(x_new,y)
list4=SKB.get_support(indices=True)
print list4
print x_new4
print x_new4.shape

#RFE特征选择
from sklearn.feature_selection import RFE
# X为样本集合，每个样本为一个数组，数组元素为各个特征值,Y样本的评分
svc=SVC(kernel='linear',C=5)
rfe=RFE(estimator=svc,n_features_to_select=14,step=1)
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


#综合一下：0.883333333333  0.15  平均训练集精度 =  0.85  平均测试集精度 =  0.80625
x_new5=x_new[:,[1,3,19,23,27,29,49,51]]
#x_new5=x_new[:,[1,3,19,27,49]]
#RFE:0.866666666667  0.163299316186  平均训练集精度 =  0.863043478261  平均测试集精度 =  0.8375
#x_new5=x_new[:,[1,3,19,20,27,29,49,51]]
print x_new5

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
    x_train, x_test, y_train, y_test = train_test_split(x_new5, y, train_size=0.75)
    model=svc.fit(x_train, y_train)
    a=svc.score(x_test, y_test)
    b=svc.score(x_train, y_train)
    #print '训练集精度 = ',b
    #print '测试集精度 = ', a
    average=average+a
    average1 = average1 + b
print '平均训练集精度 = ',average1/10
print '平均测试集精度 = ',average/10

print("done in %0.3fs" % (time.time() - t0))