# -*- coding:utf-8 -*-
import time
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
svc = SVC(kernel='linear', C=5)
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 4 ,5], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4 ,6 ,10]}
svr = SVC()
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(x_new, y)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_result.csv','w') as f:
     cv_result.to_csv(f)
print('The parameters of the best model are: ')
print(clf.best_params_)


#用cross_val_score实现10折验证，进行SVM测试，得到分类正确率
svc = SVC(kernel='rbf', C=1 ,gamma=4)
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


print("done in %0.3fs" % (time.time() - t0))
