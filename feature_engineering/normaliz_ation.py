from sklearn.preprocessing import  Normalizer
import numpy as np

# 归一化，每一行的数据和为1，，
X = np.array([
    [1, -1, 2],
    [2, 0, 0],
    [0, 1, -1]
], dtype=np.float64)

normalizer1 = Normalizer(norm='max')
normalizer2 = Normalizer(norm='l2')
normalizer1.fit(X)
normalizer2.fit(X)

print(normalizer1.transform(X))
print("----------------------------------")
print(normalizer2.transform(X))