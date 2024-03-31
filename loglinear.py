import csv
from collections import Counter
from tqdm import tqdm
import json
import numpy as np
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import string

lemmatizer = WordNetLemmatizer()
def pre_process_text(text : str):
    text = text.strip()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    res = ""
    for w in tokens:
        res += w + ' '
    return res

def Read_CSV(file_path, desc, total_count):
    csv_reader = csv.DictReader(open(file_path), fieldnames=['label', 'title', 'text'])
    data = []
    labels = []

    for row in tqdm(csv_reader, desc=desc, total=total_count):
        labels.append(int(row['label']) - 1)
        text = row['title'] + ' ' + row['text']
        filtered_words = pre_process_text(text)
        data.append(filtered_words)
    return data, labels

data, labels = Read_CSV("ag_news_csv/train.csv", "Reading train.csv", 120000)
vectorizer = CountVectorizer(max_features=20000)
transformer = TfidfTransformer()

print("Computing tf-idf ...", end=' ')
result = transformer.fit_transform(vectorizer.fit_transform(data)).astype(np.float32)
np_result = result.toarray()
labels = np.array(labels)
print("Done!")

num_epoch = 2
data_size = 120000
batch_size = 1024
feature_size = 20000
lr = 1e-1

class Data_Iterator:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.btch_siz = batch_size
    def __iter__(self):
        self.order = np.random.permutation(data_size)
        self.idx = 0
        return self
    def __len__(self):
        return int(np.ceil(data_size / self.btch_siz))
    def __next__(self):
        if self.idx == data_size:
            raise StopIteration
        nxt = min(self.idx + self.btch_siz, data_size)
        slce = self.order[self.idx : nxt]
        self.idx = nxt
        return self.data[slce].astype(np.float64), self.labels[slce]

class Log_Linear:
    def __init__(self):
        self.params = np.random.rand(4, feature_size).astype(np.float64)
    def forward(self, data_batch : np.ndarray, labels_batch : np.ndarray, require_cache=True):
        self.product = data_batch @ np.transpose(self.params)
        result = np.argmax(self.product, axis=1)
        acc = float(np.sum(result == labels_batch)) / labels_batch.shape[0]
        if require_cache == False:
            return result, acc
        self.cache = data_batch
        self.labels_cache = labels_batch
        self.sum_exp = np.sum(np.exp(self.product), axis=1)
        acc_params = self.params[labels_batch]
        loss = float(np.mean(-np.sum(data_batch * acc_params, axis=1) + np.log(self.sum_exp)))
        return result, loss, acc
    def backward(self):
        self.prob = np.exp(self.product) / np.expand_dims(self.sum_exp, axis=1)
        self.grad = np.zeros_like(self.params, dtype=np.float64)
        for i in range(self.cache.shape[0]):
            self.grad += np.outer(self.prob[i], self.cache[i])
            self.grad[self.labels_cache[i]] -= self.cache[i]

        self.params -= self.grad * lr


data_iter = Data_Iterator(np_result, labels, batch_size)
model = Log_Linear()

print("Training:")

for epoch in range(num_epoch):
    if epoch == 0:
        lr = 1e-1
    elif epoch == 1:
        lr = 1e-2
    else:
        lr = 1e-3

    with tqdm(iter(data_iter)) as pbar:
        for data_batch, labels_batch in pbar:
            _, loss, acc = model.forward(data_batch, labels_batch)
            pbar.set_description_str("epoch {}: acc = {acc:.5f}    loss = {loss:.5f}".format(epoch + 1, acc = acc, loss = loss))
            model.backward()


data, labels = Read_CSV("ag_news_csv/test.csv", "Reading test.csv", 7600)
test_vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)
test_transformer = TfidfTransformer()

print("Computing tf-idf ...", end=' ')
result = test_transformer.fit_transform(test_vectorizer.fit_transform(data)).astype(np.float32)
result_np = result.toarray()
labels = np.array(labels)
print("Done!")

print("Testing:")
result, acc = model.forward(result_np, labels, require_cache=False)

f1_score_ave = 0
for l in range(4):
    labels_l = (labels == l)
    result_l = (result == l)
    correct_count = np.dot(labels_l.astype(int), result_l.astype(int))
    precision = correct_count / np.sum(result_l)
    recall = correct_count / np.sum(labels_l)
    f1_score = 2 * recall * precision / (recall + precision)
    f1_score_ave += f1_score
    print("labels {}: precision={:.5f} recall={:.5f} F1-score={:.5f}".format(l + 1, precision, recall, f1_score))

f1_score_ave /= 4
print("Average F1-score: {:.5f}".format(f1_score_ave))