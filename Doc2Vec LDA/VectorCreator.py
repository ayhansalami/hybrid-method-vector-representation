import re
from gensim import models

from nltk.corpus import stopwords
from sklearn.datasets import load_files, twenty_newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

DOC2VEC_CLUSTERING_PATH = '/media/ayhan/My owns/master/NLP/Project/TextClusteringClassification/clustring'
MALLET_CLUSTERING_PATH = '/home/ayhan/Dropbox/nlp-paper/clustering mallet/'
MALLET_HOME_PATH = '/home/ayhan/Dropbox/nlp-paper/Mallet/'
DOC2VEC_TEST_PATH = '/media/ayhan/My owns/master/NLP/Project/TextClusteringClassification/Data/'
DOC2VEC_HOME_PATH = '/media/ayhan/My owns/master/NLP/Project/TextClusteringClassification/merged_data/'
n_features = 5000

def getTrainVectors(foldNum, vectorLength=50):
    if vectorLength == 50:
        f = open(MALLET_HOME_PATH + 'VectorsForMergedExceptFold' + foldNum + '.txt')
    elif vectorLength == 100:
        f = open(MALLET_HOME_PATH + 'VectorsForMergedExceptFold' + foldNum + '-100.txt')
    elif vectorLength == 200:
        f = open(MALLET_HOME_PATH + '200/VectorsForMergedExceptFold' + foldNum + '-200.txt')
    s = f.readlines()
    vecs = []
    for vec in s:
        vecs.append(re.split('\s+', vec.strip()))
    trainVecs = []
    for vec in vecs:
        trainVecs.append(vec[1:len(vec)])

    for i in range(len(trainVecs)):
        trainVecs[i][0] = trainVecs[i][0].replace('file:/Users/armanrahbar/Documents/master/term-2/SNLP/project/Mallet-Fold'+foldNum+'/files/','')
        trainVecs[i][0] = trainVecs[i][0].replace('.txt', '')
        trainVecs[i][0] = int(trainVecs[i][0])
        trainVecs[i] = map(float, trainVecs[i])

    trainVecs = sorted(trainVecs, key=lambda x:x[0])
    for i in range(len(trainVecs)):
        trainVecs[i] = trainVecs[i][1:len(trainVecs[i])]
    return trainVecs

def getTrainLabels(foldNum):
    f = open(MALLET_HOME_PATH+'labels-train-fold'+foldNum+'.txt')
    s = f.readlines()
    lines = []
    for vec in s:
        lines.append(re.split('\s+', vec.strip()))
    labels = []
    for i in range(len(lines)):
        labels.append(int(lines[i][1]))

    return labels



def getTestLabels(foldNum):
    f = open(MALLET_HOME_PATH+'labels-test-fold'+foldNum+'.txt')
    s = f.readlines()
    lines = []
    for vec in s:
        lines.append(re.split('\s+', vec.strip()))
    labels = []
    for i in range(len(lines)):
        labels.append(int(lines[i][1]))
    return labels

def getTestVectors(foldNum, vectorLength=50):
    if vectorLength == 50:
        f = open(MALLET_HOME_PATH + 'infered-Fold' + foldNum + '.txt')
    elif vectorLength == 100:
        f = open(MALLET_HOME_PATH + 'infered-Fold' + foldNum + '-100.txt')
    elif vectorLength == 200:
        f = open(MALLET_HOME_PATH + '200/infered-Fold' + foldNum + '-200.txt')
    s = f.readlines()
    s = s[1:len(s)]

    vecs = []
    for vec in s:
        vecs.append(re.split('\s+', vec.strip()))

    trainVecs = []
    for vec in vecs:
        trainVecs.append(vec[1:len(vec)])


    for i in range(len(trainVecs)):
        trainVecs[i][0] = trainVecs[i][0].replace(
            'file:/Users/armanrahbar/Documents/master/term-2/SNLP/project/test-mallet-fold'+foldNum+'/files/', '')
        trainVecs[i][0] = trainVecs[i][0].replace('.txt', '')
        trainVecs[i][0] = int(trainVecs[i][0])
        trainVecs[i] = map(float, trainVecs[i])

    trainVecs = sorted(trainVecs, key=lambda x: x[0])
    for i in range(len(trainVecs)):
        trainVecs[i] = trainVecs[i][1:len(trainVecs[i])]
    return trainVecs

def getDoc2vecModel(foldNum, size=50):
    dataset = load_files(container_path=DOC2VEC_HOME_PATH+'Fold_'+foldNum, shuffle=True, random_state=1, encoding='latin1')
    dataset.data = [twenty_newsgroups.strip_newsgroup_header(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_footer(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_quoting(text) for text in dataset.data]
    stop = set(stopwords.words('english'))
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '>', '<', '=', '|']
    stop.update(punctuations)
    dataToVectorize = []
    for data in dataset.data:
        dataToVectorize.append(' '.join(
            [''.join([c for c in i if c not in set(punctuations)]) for i in data.lower().split() if i not in stop]))
        dataToVectorize[-1] = dataToVectorize[-1].strip()
    doc1 = list(dataToVectorize)
    sentences = []
    for i, text in enumerate(doc1):
        sentences.append(models.doc2vec.LabeledSentence(words=text.lower().split(), tags=[str(i)]))
    model = models.Doc2Vec(alpha=.025, min_alpha=.025, size=size, negative=20, dm=2)
    model.build_vocab(sentences)
    for epoch in range(10):
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate`
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    return model

def getTfidfModel(foldNum):
    dataset = load_files(container_path=DOC2VEC_HOME_PATH+'Fold_'+foldNum, shuffle=True, random_state=1, encoding='latin1')
    dataset.data = [twenty_newsgroups.strip_newsgroup_header(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_footer(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_quoting(text) for text in dataset.data]
    stop = set(stopwords.words('english'))
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '>', '<', '=', '|']
    stop.update(punctuations)
    dataToVectorize = []
    for data in dataset.data:
        dataToVectorize.append(' '.join(
            [''.join([c for c in i if c not in set(punctuations)]) for i in data.lower().split() if i not in stop]))
        dataToVectorize[-1] = dataToVectorize[-1].strip()
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=5000,
                                   stop_words='english')
    X_train = vectorizer.fit_transform(dataset.data)
    return {
        'train': X_train,
        'model': vectorizer
    }

def getDoc2VecInferedVector(model, foldNum):
    testset = load_files(container_path=DOC2VEC_TEST_PATH+'Fold_'+foldNum, shuffle=True, random_state=1, encoding='latin1')
    testset.data = [twenty_newsgroups.strip_newsgroup_header(text) for text in testset.data]
    testset.data = [twenty_newsgroups.strip_newsgroup_footer(text) for text in testset.data]
    testset.data = [twenty_newsgroups.strip_newsgroup_quoting(text) for text in testset.data]
    stop = set(stopwords.words('english'))
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '>', '<', '=', '|']
    stop.update(punctuations)
    testToVectorize = []
    for data in testset.data:
        testToVectorize.append(' '.join(
            [''.join([c for c in i if c not in set(punctuations)]) for i in data.lower().split() if i not in stop]))
        testToVectorize[-1] = testToVectorize[-1].strip()
    result = []
    for doc in testToVectorize:
        result.append(model.infer_vector(doc.split()))
    return result

def getTfidfInferedVector(model, foldNum):
    testset = load_files(container_path=DOC2VEC_TEST_PATH+'Fold_'+foldNum, shuffle=True, random_state=1, encoding='latin1')
    testset.data = [twenty_newsgroups.strip_newsgroup_header(text) for text in testset.data]
    testset.data = [twenty_newsgroups.strip_newsgroup_footer(text) for text in testset.data]
    testset.data = [twenty_newsgroups.strip_newsgroup_quoting(text) for text in testset.data]
    stop = set(stopwords.words('english'))
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '>', '<', '=', '|']
    stop.update(punctuations)
    testToVectorize = []
    for data in testset.data:
        testToVectorize.append(' '.join(
            [''.join([c for c in i if c not in set(punctuations)]) for i in data.lower().split() if i not in stop]))
        testToVectorize[-1] = testToVectorize[-1].strip()
    return model.transform(testToVectorize)

def getLdaVecsForKmeans(vectorLength=50):
    if vectorLength == 50:
        f = open(MALLET_CLUSTERING_PATH + 'VectorsComplete.txt')
    elif vectorLength == 100:
        f = open(MALLET_CLUSTERING_PATH + 'VectorsComplete-100.txt')
    elif vectorLength == 200:
        f = open(MALLET_CLUSTERING_PATH + 'VectorsComplete-200.txt')
    s = f.readlines()
    vecs = []
    for vec in s:
        vecs.append(re.split('\s+', vec.strip()))
    trainVecs = []
    for vec in vecs:
        trainVecs.append(vec[1:len(vec)])

    for i in range(len(trainVecs)):
        trainVecs[i][0] = trainVecs[i][0].replace('file:/Users/armanrahbar/Documents/master/term-2/SNLP/project/Mallet-Complete-Corpus/files/','')
        trainVecs[i][0] = trainVecs[i][0].replace('.txt', '')
        trainVecs[i][0] = int(trainVecs[i][0])
        trainVecs[i] = map(float, trainVecs[i])

    trainVecs = sorted(trainVecs, key=lambda x:x[0])
    for i in range(len(trainVecs)):
        trainVecs[i] = trainVecs[i][1:len(trainVecs[i])]
    return trainVecs

def getLdaLabelsForKmeans():
    f = open(MALLET_CLUSTERING_PATH+'labels.txt')
    s = f.readlines()
    lines = []
    for vec in s:
        lines.append(re.split('\s+', vec.strip()))
    labels = []
    for i in range(len(lines)):
        labels.append(int(lines[i][1]))

    return labels

def getDoc2vecModelforKmeans(size=50):
    dataset = load_files(container_path=DOC2VEC_CLUSTERING_PATH, shuffle=True, random_state=1, encoding='latin1')
    dataset.data = [twenty_newsgroups.strip_newsgroup_header(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_footer(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_quoting(text) for text in dataset.data]
    stop = set(stopwords.words('english'))
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '>', '<', '=', '|']
    stop.update(punctuations)
    dataToVectorize = []
    for data in dataset.data:
        dataToVectorize.append(' '.join(
            [''.join([c for c in i if c not in set(punctuations)]) for i in data.lower().split() if i not in stop]))
        dataToVectorize[-1] = dataToVectorize[-1].strip()
    doc1 = list(dataToVectorize)
    sentences = []
    for i, text in enumerate(doc1):
        sentences.append(models.doc2vec.LabeledSentence(words=text.lower().split(), tags=[str(i)]))
    model = models.Doc2Vec(alpha=.025, min_alpha=.025, size=size, negative=20, dm=2)
    model.build_vocab(sentences)
    for epoch in range(5):
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate`
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    return model

def getTfModelForKmeans():
    dataset = load_files(container_path=DOC2VEC_CLUSTERING_PATH, shuffle=True, random_state=1, encoding='latin1')
    dataset.data = [twenty_newsgroups.strip_newsgroup_header(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_footer(text) for text in dataset.data]
    dataset.data = [twenty_newsgroups.strip_newsgroup_quoting(text) for text in dataset.data]
    stop = set(stopwords.words('english'))
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '>', '<', '=', '|']
    stop.update(punctuations)
    dataToVectorize = []
    for data in dataset.data:
        dataToVectorize.append(' '.join(
            [''.join([c for c in i if c not in set(punctuations)]) for i in data.lower().split() if i not in stop]))
        dataToVectorize[-1] = dataToVectorize[-1].strip()
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=1000,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(list(dataToVectorize))
    return tfidf