import collections
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import VectorCreator

n_features = 5000
n_topics = 12
n_top_words = 20
n_clusters = 50

labels = VectorCreator.getLdaLabelsForKmeans()
tfidf = VectorCreator.getTfModelForKmeans()
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tfidf)
centerOfClustersTfidf = kmeans.cluster_centers_
clusterOfDatasetTfidf = {}
for index, label in enumerate(kmeans.labels_):
    if label in clusterOfDatasetTfidf:
        clusterOfDatasetTfidf[label].append(index)
    else:
        clusterOfDatasetTfidf[label] = [index]
labelOfClustersTfidf = {}
for key, value in clusterOfDatasetTfidf.iteritems():
    clusterLabelCount = [labels[x] for x in value]
    counter = collections.Counter(clusterLabelCount)
    labelOfClustersTfidf[key] = counter.most_common(1)[0]

yPred = []
for pred in kmeans.labels_:
    yPred.append(labelOfClustersTfidf[pred][0])

print(confusion_matrix(labels, yPred, labels=range(0, 12)))
print(precision_score(labels, yPred, average='micro'))
print(recall_score(labels, yPred, average='micro'))
print(f1_score(labels, yPred, average='micro'))