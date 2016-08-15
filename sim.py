#coding:utf-8
import jieba
from gensim import corpora, models, similarities


documents = ["编辑心中最美中级车一汽-大众新cc",
              "25万时尚品质4款豪华紧凑车之奔驰a级 ",
               "并发中的锁文件模式"]

texts = [[word for word in jieba.cut(str(document), cut_all=False) if len(word) > 1 ] for document in documents]


# texts = [[word for word in document.lower().split()] for document in
# documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]        #tf-idf 向量

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
num_topics=2)

corpus_lsi = lsi[corpus_tfidf]        #吧向量添加到lsi里面

index = similarities.MatrixSimilarity(lsi[corpus])   #建立索引

query = "最美编辑时尚"
query_bow = dictionary.doc2bow([word for word in jieba.cut(str(query), cut_all=False) if len(word) > 1 ])
query_lsi = lsi[query_bow]    #把向量添加到lsi里面
sims = index[query_lsi]       #添加到索引
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sort_sims