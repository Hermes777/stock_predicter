#coding:utf-8
import numpy as np
from sklearn import mixture
#生成随机观测点，含有2个聚集核心
obs = np.concatenate((np.random.randn(100, 1), 10 + np.random.randn(300, 1)))
clf = mixture.GMM(n_components=2)
print obs[:10]
clf.fit(obs)
#预测
print clf.predict([[0], [2], [9], [10]])
