import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
# 三种计数方法： tf-idf  hash  count

arr1 = [
    "This is spark, spark sql a every good",
    "Spark Hadoop Hbase",
    "This is sample",
    "This is anthor example anthor example",
    "spark hbase hadoop spark hive hbase hue oozie",
    "hue oozie spark spark spark spark spark spark"
]
arr2 = [
    "this is a sample a example",
    "this c c cd is another another sample example example",
    "spark Hbase hadoop Spark hive hbase"
]
df = arr2

# tf-idf
tfidf = TfidfVectorizer(min_df=0, dtype=np.float64)
df2 = tfidf.fit_transform(df)
print(df2.toarray()) # 返回的是df 中每段文本的计算结果，，arr2中有三段文字
print(tfidf.get_feature_names())
print (tfidf.get_stop_words())
print ("转换另外的文档数据")
print (tfidf.transform(arr1).toarray())

print('*'*20)
hashing = HashingVectorizer(n_features=20, non_negative=True, norm=None)
df3 = hashing.fit_transform(df)
print (df3.toarray())
print (hashing.get_stop_words())
print ("转换另外的文档数据")
print (hashing.transform(arr1).toarray())


##
print('*'*20)
count = CountVectorizer(min_df=0.1, dtype=np.float64, ngram_range=(0,1))
df4 = count.fit_transform(df)
print (df4.toarray())
print (count.get_stop_words())
print (count.get_feature_names())
print ("转换另外的文档数据")
print (count.transform(arr1).toarray())
print(df4)