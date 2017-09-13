import collections

import numpy
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import VectorCreator

tfidfPrecisionSum = 0
tfidfRecallSum = 0
tfidfFmeasureSum = 0
classifierForTfidf = svm.SVC(kernel='rbf', C=50, gamma=.1)
for foldNum in range(1, 6):
    foldNum = str(foldNum)
    tfidfModel = VectorCreator.getTfidfModel(foldNum)
    tfidfTrain = tfidfModel['train']
    tfidfTest = VectorCreator.getTfidfInferedVector(tfidfModel['model'], foldNum)
    classifierForTfidf.fit(tfidfTrain, VectorCreator.getTrainLabels(foldNum))
    predictedList = classifierForTfidf.predict(tfidfTest).tolist()
    gold = VectorCreator.getTestLabels(foldNum)
    precision = precision_score(gold, predictedList, average='micro')
    recall = recall_score(gold, predictedList, average='micro')
    f1 = f1_score(gold, predictedList, average='micro')
    tfidfPrecisionSum += precision
    tfidfRecallSum += recall
    tfidfFmeasureSum += f1
    print('TFIDF ONLY FOR FOLD ' + foldNum)
    # print(confusion_matrix(gold, predictedList, labels=range(0, 12)))
    print(precision)
    print(recall)
    print(f1)
print('Average for lda:')
print(tfidfPrecisionSum/5)
print(tfidfRecallSum/5)
print(tfidfFmeasureSum/5)