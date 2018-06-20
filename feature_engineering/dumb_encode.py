from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import numpy as np

enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2],[1,1,1]])
a=np.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2],[1,1,1]])
print(a)
print('编码结果：', enc.n_values_)  # [2 3 4]  表示每列中 哑编码的个数

# 用哑编码对 [0, 1, 3] 进行编码
print(enc.transform([[0, 1, 3]]).toarray())

# 用的不多
# sparse：最终产生的结果是否是稀疏化矩阵，默认为True，一般不改动
dv = DictVectorizer(sparse=False)
D = [{'foo':1, 'bar':2.2}, {'foo':3, 'baz': 2}]
X = dv.fit_transform(D)
print (X)
# 直接把字典中的key作为特征，value作为特征值，然后构建特征矩阵
print (dv.get_feature_names())
print (dv.transform({'foo':4, 'unseen':3}))


h = FeatureHasher(n_features=3)
D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
# 直接以hash值计算结果 -- 该方式一般不用
f = h.transform(D)
f.toarray()
