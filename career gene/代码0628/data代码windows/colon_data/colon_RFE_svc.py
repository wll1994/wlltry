# -*- coding:utf-8 -*-
#RFE加SVM，10折验证+2折测试集准确率，选择14个，0.9和0.9375
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

np.set_printoptions(threshold=10000)  #设置打印数量的阈值
np.set_printoptions(linewidth=20000)   #使一行能放下一组数据
np.set_printoptions(suppress=True)     #取消科学计数法为正常输出

data=sio.loadmat('colon_tumor.mat')   #读取MATLAB数据,data为字典对象
data_data=data['ColonDataSet']       #此为数组，第一列为分类值后标签，第二列到最后为特征值
np.set_printoptions(linewidth=2000)   #使一行能放下一组数据
print data_data
print data_data.shape

y=data_data[:,0]  #读取标签
x=data_data[:,1:]   #读取特征值

t0 = time.time()

from sklearn import preprocessing  #预处理，特征归一化
nx = preprocessing.normalize(x)
print nx
print y

#使用卡方检验来对特征进行测试
from sklearn.feature_selection import SelectKBest,chi2
x_new=SelectKBest(chi2,k=70).fit_transform(nx,y)
print x_new
print x_new.shape

from sklearn.feature_selection import RFE   #特征选择
# X为样本集合，每个样本为一个数组，数组元素为各个特征值,Y样本的评分
svc=SVC(kernel='linear',C=5)
rfe=RFE(estimator=svc,n_features_to_select=8,step=1)
x_new2=rfe.fit_transform(x_new,y)
list2=rfe.get_support(indices=True)
print list2
print x_new2
print y

#SVM调参数
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1,5,10,50,100], 'gamma': [0.001,0.01, 0.1, 1, 10 ,100,1000]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new2, y)
print('The parameters of the best model are: ')
print(clf.best_params_)

#用SVM测试，得到分类正确率
svc=SVC(kernel='rbf', C=1,gamma=100)
#10折交叉验证
result = cross_val_score(svc, x_new2, y, cv=10)   #可设置scoring='roc_auc'，不设置时默认为准确率
print result
print result.mean()   #平均分数
print result.std()    #标准偏差估计分数

average=0
average1=0
#用train_test_split分为测试集和训练集，进行SVM测试，得到分类正确率
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x_new2, y, train_size=0.75)
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
# done in 9.541s
