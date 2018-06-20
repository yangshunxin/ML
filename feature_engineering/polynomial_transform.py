import numpy as np
from sklearn.preprocessing import  PolynomialFeatures

X = np.arange(6).reshape(3,2)
print(X)
print('====================')
poly1 = PolynomialFeatures(2)
poly1.fit(X)
print(poly1)
print(poly1.transform(X))

print('============================')
# interaction_only 只有交叉项
poly2 = PolynomialFeatures(interaction_only=True)
poly2.fit(X)
print(poly2)
print(poly2.transform(X))

print('=======================')
# 有无 特征为1 的项
poly3 = PolynomialFeatures(include_bias=False)
poly3.fit(X)
print(poly3)
print(poly3.transform(X))