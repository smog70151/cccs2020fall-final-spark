import os
import sys
import pdb
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import gensim
from gensim.models import word2vec
import numpy as np
import pandas as pd
import string
from utils.utils import Preprocess
import csv
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper

def train_word2vec(x,sg=1):
    # 訓練word to vector 的 word embedding
    print("strat training w2v")
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=sg)
    print("training finish!!!")
    return model

def main():
    # create corpus
    # Data Preprocessing
    df = pd.read_json('datas/News_Category_Dataset_v2.json', lines=True)
    df = pd.concat([df.loc[df.category == 'SPORTS'], df.loc[df.category == 'FOOD & DRINK'], df.loc[df.category == 'WOMEN']], axis=0)
    df = pd.DataFrame(np.concatenate([df[["category", "headline"]].values, df[["category", "short_description"]].values], axis=0))
    '''
        WOMEN: 0, SPORTS: 1, FOOD & DRINK: 2
    '''
    df[0].replace(["WOMEN","SPORTS","FOOD & DRINK"],[0,1,2],inplace=True)
    # Tokenzier, Stemming, Stopword
    # nltk.download('stopwords')
    # st = word_tokenize(df.iloc[0,1])
    '''
    stemmer = PorterStemmer()
    corpus = []
    f = open("corpus.txt","w")
    fl = open("labels.txt", "w")
    for index, row in df.iterrows():
        st = word_tokenize(row[1])
        new_st = []
        for word in st:
            stem_word = stemmer.stem(word)
            if stem_word not in stopwords.words('english') + [s for s in string.punctuation]:
                # print (stem_word, end=' ')
                new_st.append(stem_word)
        if len(new_st) > 2:
            row[1] = ' '.join(new_st)
            corpus.append(row[1])
            f.write(row[1]+"\n")
            fl.write(str(df[0].iloc[index])+"\n")
        else:
            continue
    f.close()
    '''
    #train w2v model
    '''
    with open('./corpus.txt', 'r') as f:
        corpus = f.readlines()
    corpus = [line.strip('\n').split(' ') for line in corpus]
    model = train_word2vec(corpus)
    model.save("w2v.model")
    '''
    #tfidf
    with open('./corpus.txt','r') as f:
        corpus = f.readlines()
    corpus = [line.strip('\n') for line in corpus]

    # ref: https://blog.csdn.net/Eastmount/article/details/50323063
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=np.array(vectorizer.get_feature_names())
    weight=tfidf.toarray()
    train_raw_x = []
    for i in range(len(weight)):
        idx = weight[i].argsort()[-3:][::-1]
        if i %100 == 0:
            print(word[idx])
        train_raw_x.append(word[idx])
    # train w2v
    preprocess = Preprocess(train_raw_x, 3, "w2v.model")
    embedding = preprocess.make_embedding(load=True)
    w2i_dict = preprocess.get_dict()

    # pDF
    fl = open('labels.txt', 'r')
    with open('./datas/trains.csv', 'w', newline='') as csvfile:
        # writer = csv.writer(csvfile)
        temp = []
        for index, raw in enumerate(train_raw_x):
            label = np.array([int(fl.readline().strip('\n'))])
            raw = preprocess.sentence_word2idx(raw)
            train = np.concatenate([label, embedding[raw[0]], embedding[raw[1]], embedding[raw[2]]], axis=0)
            temp.append(train)
        np.savetxt('./datas/trains.csv', temp, delimiter=',')

    spark = SparkSession.builder \
        .appName("examples") \
        .master('local[2]').config('spark.driver.memory', '2g') \
        .getOrCreate()

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('datas/trains.csv').orderBy(rand()).repartition(2)
    # rdd = SQLContext.createDataFrame(pDF).rdd
    network = nn.Sequential(
        nn.Linear(750, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
    )

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.001
    )

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:751], outputCol='features')

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=50,
        verbose=1,
        miniBatch=256,
        earlyStopPatience=40,
        validationPct=0.2
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
    p.write().overwrite().save('simple_nn')

    # Example of loading the pipeline
    loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_nn'))

    # Run predictions and evaluation
    predictions = loaded_pipeline.transform(df).persist()
    evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Train accuracy = %g" % accuracy)

if __name__ == '__main__':
    main()
