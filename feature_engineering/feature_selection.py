import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_regression, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier

X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0.1, 1, 1, 3]
], dtype=np.int32)
Y = np.array([1,2,1])

## 删除 方差 小于0.1 的特征
variance = VarianceThreshold(threshold=0.1)
print(variance)
variance.fit(X)
print(variance.transform(X))

# 相关系数法，
sk1 = SelectKBest(f_regression, k=2) # k表示需要选择的特征个数
sk1.fit(X, Y)
print(sk1)
print(sk1.scores_)
print(sk1.transform(X))

# 卡方检验 ： 用来检验两个特征的相关关系 --------------实际工作中用的多
sk2 = SelectKBest(chi2, k=2)
sk2.fit(X, Y)
print(sk2)
print(sk2.scores_)
print(sk2.transform(X))

# print('==============================')
# 递归特征消除法
estimator = SVR(kernel='linear')  # 支持向量机（SVM） 线性核函数
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.n_features_)  # 2
print(selector.ranking_)
print(selector.transform(X))


#  线性回归中的L1正则
X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2],
    [ 4.9,  3. ,  1.4,  0.2],
    [ -6.2,  0.4,  5.4,  2.3],
    [ -5.9,  0. ,  5.1,  1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
estimator = LogisticRegression(penalty='l1', C=0.1)#  LASSO回归，
sfm = SelectFromModel(estimator)
sfm.fit(X2, Y2)
print(sfm.transform(X2))  # 只选出了一个？？？？？？？


# GBDT
X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2],
    [ 4.9,  3. ,  1.4,  0.2],
    [ -6.2,  0.4,  5.4,  2.3],
    [ -5.9,  0. ,  5.1,  1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
estimator = GradientBoostingClassifier()
sfm = SelectFromModel(estimator)
sfm.fit(X2, Y2)
print(sfm.transform(X2))  # 选出了两个特征

##  PCA降维
from sklearn.decomposition import PCA
X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2,1,23],
    [ 4.9,  3. ,  1.4,  0.2,2.3,2.1],
    [ -6.2,  0.4,  5.4,  2.3,2,23],
    [ -5.9,  0. ,  5.1,  1.8,2,3]
], dtype=np.float64)
pca = PCA(n_components=1)
pca.fit(X2) # X2是4*6的矩阵
print(pca.mean_)
print(pca.components_)
print(pca.transform(X2))

# LDA 降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2],
    [ 4.9,  3. ,  1.4,  0.2],
    [ -6.2,  0.4,  5.4,  2.3],
    [ -5.9,  0. ,  5.1,  1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X2, Y2)
print(lda.transform(X2))


## LDA sample 2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.array([
    [-1, -1],
    [-2, -1],
    [-3, -2],
    [1, 1],
    [2, 1],
    [3, 2]])
y = np.array([1, 1, 2, 2, 1, 1])
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)


print(clf.predict([[-0.8, -1]]))
clf.coef_
