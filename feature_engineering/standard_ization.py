from sklearn.preprocessing import StandardScaler

X = [
    [1, 2, 3, 2],
    [7, 8, 9, 2.01],
    [4, 8, 2, 2.01],
    [9, 5, 2, 1.99],
    [7, 5, 3, 1.99],
    [1, 4, 9, 2]
]

ss = StandardScaler(with_mean=True, with_std=True)
ss.fit(X)

print(ss.mean_) # 均值
print(ss.n_samples_seen_) #样本数
print(ss.scale_)#

print(ss.transform(X))