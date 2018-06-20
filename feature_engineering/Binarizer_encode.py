import numpy as np
from sklearn.preprocessing import Binarizer

arr = np.array([
    [1.5, 2.3, 1.9],
    [0.5, 0.5, 1.6],
    [1.1, 2, 0.2]
])

binarizer = Binarizer(threshold=1.0).fit(arr)
print(binarizer.transform(arr))