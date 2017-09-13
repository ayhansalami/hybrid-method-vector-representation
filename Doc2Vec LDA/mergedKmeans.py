from __future__ import print_function
from nltk.corpus import stopwords
from gensim import models

# Import libraries

from gensim.models import doc2vec
from collections import namedtuple

# Load data

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from time import time

import collections
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files, twenty_newsgroups
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import CompareClusterings
import VectorCreator

n_features = 5000
n_topics = 12
n_top_words = 20
n_clusters = 50

labels = VectorCreator.getLdaLabelsForKmeans()
doc2vecVectors = VectorCreator.getDoc2vecModelforKmeans()
ldaVectors = VectorCreator.getLdaVecsForKmeans()
# test_doc_topic_dist_unnormalized = np.matrix(lda.transform(testsetTfVector)).tolist()
merged = []
for i, l in enumerate(doc2vecVectors.docvecs):
    merged.append([x for x in l]+[y for y in ldaVectors[i]])

print("MERGED RESULT:")
for i in range(0, 3):
    max = []
    mean = []
    for idx, val in enumerate(merged[0]):
        itemMax = 0
        itemSum = 0
        for item in merged:
            if item[idx] > itemMax:
                itemMax = item[idx]
            itemSum += item[idx]
        max.append(itemMax)
        mean.append(itemSum / len(merged))
    for tfIdx, item in enumerate(merged):
        for fieldIdx, field in enumerate(item):
            merged[tfIdx][fieldIdx] = (field - mean[fieldIdx]) / max[fieldIdx]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(list(merged))
clusterOfDataset = {}
for index, label in enumerate(kmeans.labels_):
    if label in clusterOfDataset:
        clusterOfDataset[label].append(index)
    else:
        clusterOfDataset[label] = [index]

labelOfClusters = {}
for key, value in clusterOfDataset.iteritems():
    clusterLabelCount = [labels[x] for x in value]
    counter = collections.Counter(clusterLabelCount)
    labelOfClusters[key] = counter.most_common(1)[0]

yPred = []
for pred in kmeans.labels_:
    yPred.append(labelOfClusters[pred][0])

print(confusion_matrix(labels, yPred, labels=range(0, 12)))
print(precision_score(labels, yPred, average='micro'))
print(recall_score(labels, yPred, average='micro'))
print(f1_score(labels, yPred, average='micro'))

#DOC2VEC CLUSTERING
print('DOC2VEC ONLY:')
doc2vecVectors = VectorCreator.getDoc2vecModelforKmeans(100)
ldaVectors = VectorCreator.getLdaVecsForKmeans(100)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(list(doc2vecVectors.docvecs))
centerOfClustersDoc2Vec = kmeans.cluster_centers_
clusterOfDatasetDoc2Vec = {}
for index, label in enumerate(kmeans.labels_):
    if label in clusterOfDatasetDoc2Vec:
        clusterOfDatasetDoc2Vec[label].append(index)
    else:
        clusterOfDatasetDoc2Vec[label] = [index]

labelOfClustersDoc2Vec = {}
for key, value in clusterOfDatasetDoc2Vec.iteritems():
    clusterLabelCount = [labels[x] for x in value]
    counter = collections.Counter(clusterLabelCount)
    labelOfClustersDoc2Vec[key] = counter.most_common(1)[0]

yPred = []
for pred in kmeans.labels_:
    yPred.append(labelOfClustersDoc2Vec[pred][0])

print(confusion_matrix(labels, yPred, labels=range(0, 12)))
print(precision_score(labels, yPred, average='micro'))
print(recall_score(labels, yPred, average='micro'))
print(f1_score(labels, yPred, average='micro'))

#LDA CLUSTERING
print('LDA ONLY:')
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ldaVectors)
centerOfClustersLda = kmeans.cluster_centers_
# predictions = kmeans.predict(test_doc_topic_dist_unnormalized)
# print(predictions)
clusterOfDatasetLda = {}
for index, label in enumerate(kmeans.labels_):
    if label in clusterOfDatasetLda:
        clusterOfDatasetLda[label].append(index)
    else:
        clusterOfDatasetLda[label] = [index]

labelOfClustersLda = {}
for key, value in clusterOfDatasetLda.iteritems():
    clusterLabelCount = [labels[x] for x in value]
    counter = collections.Counter(clusterLabelCount)
    labelOfClustersLda[key] = counter.most_common(1)[0]

yPred = []
for pred in kmeans.labels_:
    yPred.append(labelOfClustersLda[pred][0])

print(confusion_matrix(labels, yPred, labels=range(0, 12)))
print(precision_score(labels, yPred, average='micro'))
print(recall_score(labels, yPred, average='micro'))
print(f1_score(labels, yPred, average='micro'))

# OVERLAPPING
# print(centerOfClustersDoc2Vec)
# print(centerOfClustersLda)
# CompareClusterings.saveResultInFile(clusterOfDatasetLda, CompareClusterings.LDA_CLUSTERS)
# CompareClusterings.saveResultInFile(clusterOfDatasetDoc2Vec, CompareClusterings.DOC_2VEC_CLUSTERS)
# maxOverlapping = {}
# for key,value in clusterOfDatasetLda.iteritems():
#     for doc2vecKey, doc2vecValue in clusterOfDatasetDoc2Vec.iteritems():
#         intersection = set(value).intersection(set(doc2vecValue))
#         if key in maxOverlapping:
#             if len(intersection) > maxOverlapping[key]['intersectionLength']:
#                 maxOverlapping[key] = {
#                     'intersectionLength': len(intersection),
#                     'id': doc2vecKey,
#                     'doc2VecLength': len(doc2vecValue),
#                     'ldaLength': len(value)
#                 }
#         else:
#             maxOverlapping[key] = {
#                 'intersectionLength': len(intersection),
#                 'id': doc2vecKey,
#                 'doc2VecLength': len(doc2vecValue),
#                 'ldaLength': len(value)
#             }
#
# print(maxOverlapping)
# # OVERLAP WITH DISTANCE VECTOR
# minDistances = {}
# for key, value in enumerate(centerOfClustersLda):
#     for doc2vecKey, doc2vecValue in enumerate(centerOfClustersDoc2Vec):
#         distance = np.math.sqrt(sum([(a - b) ** 2 for a, b in zip(value, doc2vecValue)]))
#         if key in minDistances:
#             if distance < minDistances[key]['distance']:
#                 maxOverlapping[key] = {
#                     'distance': distance,
#                     'id': doc2vecKey
#                 }
#         else:
#             minDistances[key] = {
#                 'distance': distance,
#                 'id': doc2vecKey
#             }
# print(minDistances)