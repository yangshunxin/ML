import numpy as np
from sklearn.preprocessing import Imputer

X = [
    [2, 2, 4, 1],
    [np.nan, 3, 4, 4],
    [1, 1, 1, np.nan],
    [2, 2, np.nan, 3]
]
X2 = [
    [2, 6, np.nan, 1],
    [np.nan, 5, np.nan, 1],
    [4, 1, np.nan, 5],
    [np.nan, np.nan, np.nan, 1]
]

# 按照列进行填充值的计算
imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
# 按照行进行填充值
imp2 = Imputer(missing_values='NaN', strategy='mean', axis=1)

imp1.fit(X)
imp2.fit(X)
print(imp1.transform(X2))  # 用X中每列的均值 来 填充X2对应列中的缺省值，
print('--------------------')
print(imp2.transform(X2))
print(X2)

imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp2 = Imputer(missing_values='NaN', strategy='median', axis=0)
imp3 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)  #最大频率
imp1.fit(X)
imp2.fit(X)
imp3.fit(X)

print (X2)
print ("--------------------------")
print (imp1.transform(X2))
print ("--------------------------")
print (imp2.transform(X2))
print ("--------------------------")
print (imp3.transform(X2))