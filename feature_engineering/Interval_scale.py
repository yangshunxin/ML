import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([
    [1, -1, 2, 3],
    [2, 0, 0, 3],
    [0, 1, -1, 3]
], dtype=np.float64)

scaler = MinMaxScaler(feature_range=(1,5))  # 把数据都映射到（1，5）中
scaler.fit(X)

print(scaler.data_max_)
print(scaler.data_min_)
print(scaler.data_range_)

print(scaler.transform(X))

