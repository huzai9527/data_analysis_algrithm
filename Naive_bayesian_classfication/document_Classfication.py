# encoding=utf-8
import jieba
import os
import io
stop_words = [line.strip() for line in io.open("data/stop/stopword.txt", "rb").readlines()]
label_dic = {"体育": 1, "女性": 2, "文学": 3, "校园": 4}


def load_data(data_path):
    labels = []
    document = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            label = root.split("\\")[-1]
            labels.append(label_dic[label])
            filename = os.path.join(root, file)
            with open(filename, "rb") as f:
                content = f.read()
                word_list = list(jieba.cut(content))
                words = [wd for wd in word_list]
                document.append(' '.join(words))
    return document, labels


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB


def train(documents, labels):
    # 创建TfidfVectorizer,加载停用词，对于超过半数文章中出现的单词不做统计
    tfidf_vec = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    # 对Document进行拟合,得到TF-IDF矩阵
    features = tfidf_vec.fit_transform(documents)
    train_vocabulary = tfidf_vec.vocabulary_
    clf = MultinomialNB(alpha=0.001).fit(features, labels)
    return train_vocabulary, clf


def predict(train_vocabulary, clf, document):
    test_tfidf = TfidfVectorizer(stop_words=stop_words, max_df=0.5, vocabulary=train_vocabulary)
    test_features = test_tfidf.fit_transform(document)
    predict_labels = clf.predict(test_features)
    return predict_labels


train_document, train_labels = load_data("data/train")
test_document, test_labels = load_data("data/test")
train_vocabulary, clf = train(train_document, train_labels)
predict_labels = predict(train_vocabulary, clf, test_document)
print(predict_labels)
print(test_labels)
from sklearn import metrics
x = metrics.accuracy_score(test_labels, predict_labels)
print(x)








