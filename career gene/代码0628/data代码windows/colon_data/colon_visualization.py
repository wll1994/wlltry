# -*- coding:utf-8 -*-
#chi2+svm,可视化2维基因。
import time
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split   说明这个版本很早0.18以前
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

#from sklearn.preprocessing import  StandardScaler
#nx =StandardScaler().fit_transform(x)   #预处理，特征标准化，标准化结果有负数

#from sklearn.preprocessing import MinMaxScaler
#mx=MinMaxScaler().fit_transform(nx)     #预处理，区间缩放法
#print mx

#使用卡方检验来对特征进行测试
from sklearn.feature_selection import SelectKBest,chi2
x_new=SelectKBest(chi2,k=30).fit_transform(nx,y)
print x_new
print x_new.shape

#将数据分为训练集和测试集得到准确率,stratify= 0怎么用
x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75,random_state=0)
print x_train, y_train
print x_train.shape, y_train.shape
print x_test, y_test
print x_test.shape, y_test.shape
svc=SVC(kernel='linear', C=23)
model=svc.fit(x_train, y_train)
print '训练集精度 = ', svc.score(x_train, y_train)
print '测试集精度 = ', svc.score(x_test, y_test)

print("done in %0.3fs" % (time.time() - t0))
# 训练集精度 =  0.847826086957  测试集精度 =  0.9375  done in 0.011s
# 不处理，训练集精度 =  1.0   测试集精度 =  0.75

# 可视化
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

plt.figure(facecolor='w')
x1_min = x_new[0].min()
x2_min = x_new[1].min()
x1_max = x_new[0].max()
x2_max = x_new[1].max()
#x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]     # 生成网格采样点
#plt.pcolormesh(x1, x2, cmap=cm_light)
plt.scatter(x_new[:, 0], x_new[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)# 样本的显示
plt.xlabel('gene1', fontsize=13)
plt.ylabel('gene2', fontsize=13)
plt.xlim(x1_min, x1_max)  #当前的X轴绘图范围
plt.ylim(x2_min, x2_max)
plt.title(u'gene binary classification', fontsize=16)
plt.show()