import os
import csv
import sys
import pdb
import numpy as np
import pandas as pd
import gensim
from gensim.models import word2vec
import string
# Local File
from utils.utils import Preprocess
# Package - nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Package - sklearn
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# Package - torch
import torch
import torch.nn as nn
# Package - Pyspark
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper

class sparkAD:
    def __init__(self):
        self.threshold = 2
        self.spark = SparkSession.builder.appName("examples").master('local[2]').config('spark.driver.memory', '3g').getOrCreate()
        self.network = nn.Sequential(
            nn.Linear(750, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.torch_obj = serialize_torch_obj(
            model=self.network,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            lr=0.001
        )

        self.spark_model = SparkTorch(
            inputCol='features',
            labelCol='_c0',
            predictionCol='predictions',
            torchObj=self.torch_obj,
            iters=50,
            verbose=1,
            miniBatch=256,
            earlyStopPatience=40,
            validationPct=0.2
        )

        self.model_pipe = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_nn'))

    def preprocess(self, msgs):
        # self.spDF = sqlContext.createDataFrame(pd.DataFrame(np.array(), columns=["_c"+str(i) for i in range(751)]))
        self.pDF = pd.DataFrame.from_dict(msgs, orient='index')
        self.prep_nltk()
        self.prep_tfidf()
        self.get_trainX()
        res = self.inference()
        return res

    def prep_nltk(self):
        # Token, Stem, Stopwords
        stemmer = PorterStemmer()
        corpus = []
        f = open('corpus.txt', 'a')
        for index, row in self.pDF.iterrows():
            st = word_tokenize(row[0])
            new_st = []
            for word in st:
                stem_word = stemmer.stem(word)
                if stem_word not in stopwords.words('english') + [punc for punc in string.punctuation]:
                    new_st.append(stem_word)
            if len(new_st) > 2:
                row[0] = ' '.join(new_st)
                corpus.append(row[0])
                f.write(row[0]+'\n')
        f.close()

    def prep_tfidf(self):
        # TF-IDF
        with open('./corpus.txt', 'r') as f:
            corpus = f.readlines()
        corpus = [line.strip('\n') for line in corpus]
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        word = np.array(vectorizer.get_feature_names())
        weights = tfidf.toarray()
        self.train_X = []
        for i in range((len(weights)-self.threshold), len(weights), 1):
            idx = weights[i].argsort()[-3:][::-1]
            if i % 100 == 0:
                print (word[idx])
            self.train_X.append(word[idx])
        self.preprocess = Preprocess(self.train_X, 3, "w2v.model")
        self.embedding = self.preprocess.make_embedding(load=True)

    def get_trainX(self):
        self.trainX = []
        for index, raw in enumerate(self.train_X):
            raw = self.preprocess.sentence_word2idx(raw)
            train = np.concatenate([np.array([0]), self.embedding[raw[0]], self.embedding[raw[1]], self.embedding[raw[2]]], axis=0)
            self.trainX.append(train)
        self.sqlContext = SQLContext(self.spark)
        self.spDF = self.sqlContext.createDataFrame(pd.DataFrame(np.array(self.trainX), columns=["_c"+str(i) for i in range(751)]))

    def inference(self):
        outcome = [0 for i in range(3)]

        self.vector_assembler = vector_assembler = VectorAssembler(inputCols=self.spDF.columns[1:751], outputCol='features')

        predictions = self.model_pipe.transform(self.spDF).persist()
        pred_col = predictions.select("predictions").collect()

        for i in range(self.threshold):
            index = int(pred_col[i].predictions)
            print (index)
            outcome[index] = outcome[index] + 1
        res = {}
        res['prediction'] = outcome.index(max(outcome))
        print (res)

        return res

# For Testing
ad = sparkAD()
msgs = {'0': 'Hello, John Worse Temp', '1': 'Temporary Data Collapse'}
out = ad.preprocess(msgs)
