from __future__ import print_function

import numpy
import itertools
from sklearn import svm
import VectorCreator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from multisvm import MulticlassSVM

classifierForLda = svm.SVC(kernel='rbf', C=50, gamma=.1)
classifierForDoc2vec = MulticlassSVM(C=0.001, tol=0.01, max_iter=2000, random_state=0, verbose=1)
classifierForMerged = MulticlassSVM(C=0.1, tol=0.01, max_iter=2000, random_state=0, verbose=1)

ldaPrecisionSum = 0
ldaRecallSum = 0
ldaFmeasureSum = 0
doc2VecPrecisionSum = 0
doc2VecRecallSum = 0
doc2VecFmeasureSum = 0
mergedPrecisionSum = 0
mergedRecallSum = 0
mergedFmeasureSum = 0
for foldNum in range(1, 6):
    foldNum = str(foldNum)
    ldaTrainVectors = VectorCreator.getTrainVectors(foldNum, 100)
    ldaTestVectors = VectorCreator.getTestVectors(foldNum, 100)
    doc2VecModel = VectorCreator.getDoc2vecModel(foldNum, 100)
    doc2VecTrainVectors = doc2VecModel.docvecs
    doc2VecTestVectors = VectorCreator.getDoc2VecInferedVector(doc2VecModel, foldNum)
    mergedTrainVectors = []
    mergedTestVectors = []
    for idx, item in enumerate(ldaTrainVectors):
        mergedTrainVectors.append([x for x in doc2VecTrainVectors[idx]] + [y for y in item])
    for idx, item in enumerate(ldaTestVectors):
        mergedTestVectors.append([x for x in doc2VecTestVectors[idx]] + [y for y in item])
    # LDA ONLY SVM
    print(ldaTrainVectors[0])
    print(doc2VecTrainVectors[0])
    print(mergedTrainVectors[0])
    ldaTrainVectors = VectorCreator.getTrainVectors(foldNum, 200)
    ldaTestVectors = VectorCreator.getTestVectors(foldNum, 200)
    classifierForLda.fit(numpy.array(ldaTrainVectors), VectorCreator.getTrainLabels(foldNum))
    predictedList = classifierForLda.predict(ldaTestVectors).tolist()
    gold = VectorCreator.getTestLabels(foldNum)
    precision = precision_score(gold, predictedList, average='micro')
    recall = recall_score(gold, predictedList, average='micro')
    f1 = f1_score(gold, predictedList, average='micro')
    ldaPrecisionSum += precision
    ldaRecallSum += recall
    ldaFmeasureSum += f1
    print('LDA ONLY FOR FOLD '+foldNum)
    # print(confusion_matrix(gold, predictedList, labels=range(0, 12)))
    print(precision)
    print(recall)
    print(f1)
    # DOC2VEC ONLY SVM
    doc2VecModel = VectorCreator.getDoc2vecModel(foldNum, 200)
    doc2VecTrainVectors = doc2VecModel.docvecs
    doc2VecTestVectors = VectorCreator.getDoc2VecInferedVector(doc2VecModel, foldNum)
    print('DOC2VEC full length:')
    print(len(doc2VecTrainVectors[0]))
    print(len(doc2VecTestVectors[0]))
    classifierForDoc2vec.fit(numpy.array(doc2VecTrainVectors), VectorCreator.getTrainLabels(foldNum))
    predictedList = classifierForDoc2vec.predict(doc2VecTestVectors).tolist()
    gold = VectorCreator.getTestLabels(foldNum)
    precision = precision_score(gold, predictedList, average='micro')
    recall = recall_score(gold, predictedList, average='micro')
    f1 = f1_score(gold, predictedList, average='micro')
    doc2VecPrecisionSum += precision
    doc2VecRecallSum += recall
    doc2VecFmeasureSum += f1
    print('DOC2VEC ONLY FOR FOLD ' + foldNum)
    # print(confusion_matrix(gold, predictedList, labels=range(0, 12)))
    print(precision)
    print(recall)
    print(f1)
    # MERGED SVM
    # NORMALIZE VECTORS
    for i in range(0, 3):
        max = []
        mean = []
        for idx, val in enumerate(mergedTrainVectors[0]):
            itemMax = 0
            itemSum = 0
            for item in mergedTrainVectors:
                if item[idx] > itemMax:
                    itemMax = item[idx]
                itemSum += item[idx]
            max.append(itemMax)
            mean.append(itemSum / len(mergedTrainVectors))

        for tfIdx, item in enumerate(mergedTrainVectors):
            for fieldIdx, field in enumerate(item):
                mergedTrainVectors[tfIdx][fieldIdx] = (field - mean[fieldIdx]) / max[fieldIdx]
        for tfIdx, item in enumerate(mergedTestVectors):
            for fieldIdx, field in enumerate(item):
                mergedTestVectors[tfIdx][fieldIdx] = (field - mean[fieldIdx]) / max[fieldIdx]
    classifierForMerged.fit(numpy.array(mergedTrainVectors), VectorCreator.getTrainLabels(foldNum))
    predictedList = classifierForMerged.predict(mergedTestVectors).tolist()
    gold = VectorCreator.getTestLabels(foldNum)
    precision = precision_score(gold, predictedList, average='micro')
    recall = recall_score(gold, predictedList, average='micro')
    f1 = f1_score(gold, predictedList, average='micro')
    mergedPrecisionSum += precision
    mergedRecallSum += recall
    mergedFmeasureSum += f1
    print('MERGED FOR FOLD ' + foldNum)
    # print(confusion_matrix(gold, predictedList, labels=range(0, 12)))
    print(precision)
    print(recall)
    print(f1)
print('Average for lda:')
print(ldaPrecisionSum/5)
print(ldaRecallSum/5)
print(ldaFmeasureSum/5)

print('Average for doc2vec:')
print(doc2VecPrecisionSum/5)
print(doc2VecRecallSum/5)
print(doc2VecFmeasureSum/5)

print('Average for merged:')
print(mergedPrecisionSum/5)
print(mergedRecallSum/5)
print(mergedFmeasureSum/5)
