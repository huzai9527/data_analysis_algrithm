# 朴素贝叶斯
## 一、概念 
- 先验概率： 通过经验来判断事物发生的概率 
- 后验概率： 结果已经有了，推测原因的概率 - 条件概率： p(A|B) 事件A在事件B已经发生的情况下发生的概率 
- 似然函数： 用来衡量概率模型的参数
## 二、朴素贝叶斯公式理解 - 公式
![1.png](https://i.loli.net/2019/03/10/5c847adf87a93.png)
 - 对于公式的理解 ：
 - 公式左边是我们通过特征A出现，推断它属于Bi的概率
 - 公式右边分子是分类Bi在所有类别的概率*分类Bi中A出现的概率   
 - 公式右边分母是A在每个分类的概率之和
  ## 三、朴素贝叶斯分类器的工作流程
 - 确定特征属性-----------------------获取训练样本（准备阶段）
 - 计算每个类别的概率-----------------p(Ci)
  - 计算每个特征在每个类别出现的概率----p(Ai|Ci)p(Ci)
 ## 四、pyhton实现对新闻的分类（sklearn 机器学习包） 
 ### 4.1分类器的种类
  - 高斯朴素贝叶斯：特征变量是连续变量，符合高斯分布 eg：人的身高、体重 
  - 多项式朴素贝叶斯：特征变量符合多项分布 eg：单词词频(TF-IDF)  
  - 伯努利朴素贝叶斯:特征变量符合0/1分布 eg：单词是否出现
 ### 4.2概念 
- TF(Term Frequency):单词在文档中出现的次数，默认单词的重要性和它出现的次数成正比 -
- IDF(Iverse Document Frequency):单词的区分度，默认一个单词出现在的文档数越少，就越能通过这个单词把该文档和其他文档区分开
 ### 4.3如何计算TF-IDF
- 公式
-![2.png](https://i.loli.net/2019/03/10/5c847adf879c3.png)
### 4.4 sklearn 求 TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
# 创建TfidfVectorizer,加载停用词，对于超过半数文章中出现的单词不做统计
tfidf_vec = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
# 对Document进行拟合,得到各个文本各个词的TF-IDF值（分类器用到的特征空间）
features = tfidf_vec.fit_transform(documents)
```

- 注：调用后的tfidf_vec属性值
 - vocabulary_ 词汇表
 - idf_              idf值
  - stopwords_  停用词表
### 4.5 如何对文档进行分类
#### 4.5.1处理流程
![30.png](https://i.loli.net/2019/03/10/5c847adf901cb.png)
#### 4.5.2对文档进行分词
- 对于英文文档，使用NTLK包
```python
word_list = nltk.word_tokenize(text)
nltk.post_tag(word_list)
```
 - 对于中文文档，使用jieba包
 ```python
 word_list = jieba.cut(text)
 ```
 #### 4.5.3加载停用词表
 ```python
 stop_words = [line.strip().decode(utf-8) for line in io.open("stop_wordss.txt").readlines()]
 ```
#### 4.5.4 计算单词的权重
```python 
# 创建TfidfVectorizer,加载停用词，对于超过半数文章中出现的单词不做统计
tfidf_vec = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
# 对Document进行拟合,得到TF-IDF矩阵
features = tfidf_vec.fit_transform(documents)
```
#### 4.5.5 生成朴素贝叶斯分类器
- 将特征训练集的特征空间train_features,以及训练集对的train_lable传递给贝叶斯分类器
- 使用多项式分类器，alpha为平滑参数，当aplha在[0,1]之间是使用的是Lidstone平滑，当alpha=1时使用的时Laplace平滑，对于Lidstone平滑，alpha越小，迭代次数越多，精度越高
```python
from sklearn.naive_bayes import  MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(features, labels
```
#### 4.5.6 使用生成的分类器做预测- 得到测试集的特征矩阵
```python
test_tfidf_vec = TfidfVectorizer(stop_words=stop_words, max_df=0.5,vocabulary=train_vocabulary)
test_features = tfidf_vec.fit_transform(documents)
```
- 用训练好的分类器做预测
```pyhton
predict = clf.predict(test_features)
```
#### 4.5.6 计算准确率
```python
from sklearn import metrics
accuracy = metrics.accuracy_score(test_labels, predict_labels)
```
## 五、完整代码
```python
# encoding=utf-8
import jieba
import os
import io
stop_words = [line.strip() for line in io.open("data/stop/stopword.txt", "rb").readlines()]
label_dic = {"体育": 1, "女性": 2, "文学": 3, "校园": 4}

# 加载目录下的文档，返回分词后的文档和文档标签
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

# 传入分词后的文档和对应的标签，返回用于区分文档的单词表和分类器
def train(documents, labels):
    # 创建TfidfVectorizer,加载停用词，对于超过半数文章中出现的单词不做统计
    tfidf_vec = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    # 对Document进行拟合,得到各个文本各个词的TF-IDF值（分类器用到的特征空间）
    features = tfidf_vec.fit_transform(documents)
    train_vocabulary = tfidf_vec.vocabulary_
    clf = MultinomialNB(alpha=0.001).fit(features, labels)
    return train_vocabulary, clf

# 传入用于分类的单词表、分类器、以及需要预测的文档
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

```
- 注：完整代码包括数据已上传github https://github.com/huzai9527/data_analysis_algrithm
