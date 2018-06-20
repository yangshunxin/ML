import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

path = "./datas/car.data"
data = pd.read_csv(path, header=None)
print(type(data))

for i in range(7):
    print(i, np.unique(data[i]))
print(data.head(5))

### 字符串转换为序列id（数字）
X = data.apply(lambda x: pd.Categorical(x).codes)
print(X.head(5))

### 进行哑编码操作
enc = OneHotEncoder()
X = enc.fit_transform(X)
print(enc.n_values_)

### 转换后数据
df2 = pd.DataFrame(X.toarray())
print(df2.head(5))


print(df2.info())

