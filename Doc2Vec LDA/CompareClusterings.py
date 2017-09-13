from __future__ import division
import pickle

import itertools
from scipy.misc import comb
import scipy

LDA_CLUSTERS = 'lda_clusters.pk'
DOC_2VEC_CLUSTERS = 'doc2vec_clusters.pk'
COMPARE_PARAMS = 'compare_params.pk'

def saveResultInFile(object, fileName):
    with open(fileName, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

def getResultFromFile(fileName):
    with open(fileName, 'rb') as input:
        result = pickle.load(input)
    return result

ldaClusters = getResultFromFile(LDA_CLUSTERS)
doc2vecClusters = getResultFromFile(DOC_2VEC_CLUSTERS)
# FIRST METHOD
pairOfLda = []
pairOfDoc2vec = []
S = {}
allDocs = []
allPairs = []

for key, value in ldaClusters.iteritems():
    allDocs = allDocs + value
    pairOfLda.extend(list(set(itertools.combinations(value, 2))))

for key, value in doc2vecClusters.iteritems():
    pairOfDoc2vec.extend(list(set(itertools.combinations(value, 2))))

S['11'] = set(pairOfLda).intersection(pairOfDoc2vec)
S['10'] = set(pairOfLda).difference(set(pairOfDoc2vec))
S['01'] = set(pairOfDoc2vec).difference(set(pairOfLda))

print(len(S['11'])/(len(S['11'])+len(S['10'])+len(S['01'])))

m = {}
t1 = 0
t2 = 0
t3 = 0
allDocsLen = 0
result = 0

for key, value in ldaClusters.iteritems():
    t1 += comb(len(value), 2)
    allDocsLen = allDocsLen + len(value)
    for doc2vecKey, doc2vecValue in doc2vecClusters.iteritems():
        if key == 1:
            t2 += comb(len(doc2vecValue), 2)
        if key not in m:
            m[key] = {}
        m[key][doc2vecKey] = len(set(value).intersection(set(doc2vecValue)))
t3 = (2*t1*t2)/(allDocsLen*(allDocsLen-1))

for key, value in ldaClusters.iteritems():
    for doc2vecKey, doc2vecValue in doc2vecClusters.iteritems():
        result += comb(m[key][doc2vecKey], 2)

result -= t3
result /= .5*(t1+t2)-t3

print(result)