# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = [u"我 来到 北京 清华大学", 
          u"他 来到 了 网易 杭研 大厦",
          u"小明 硕士 毕业 与 中国 科学院",
          u"我 爱 北京 天安门"]

vectorizer = CountVectorizer()
transformer= TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

word = vectorizer.get_feature_names()
weight = tfidf.toarray()

for i in range(len(weight)):
    print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
    for j in range(len(word)):
        print word[j], weight[i][j]

 
