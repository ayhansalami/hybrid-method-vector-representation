import shutil
from os import listdir, makedirs, path

from os.path import isfile, join

DATASET_PATH = '/media/ayhan/My owns/master/NLP/Project/TextClusteringClassification/Data'
MERGED_DATA_PATH = '/media/ayhan/My owns/master/NLP/Project/TextClusteringClassification/merged_data'

folds = [f for f in listdir(DATASET_PATH)]
categories = [c for c in listdir(join(DATASET_PATH, folds[0]))]


for fold in folds:
    foldsToMerge = folds[:]
    foldsToMerge.remove(fold)
    makedirs(join(MERGED_DATA_PATH, fold))
    for foldToMerge in foldsToMerge:
        for category in categories:
            if not path.isdir(join(MERGED_DATA_PATH, fold, category)):
                makedirs(join(MERGED_DATA_PATH, fold, category))
            filePath = join(DATASET_PATH, foldToMerge, category)
            files = [f for f in listdir(filePath) if isfile(join(filePath, f))]
            for file in files:
                shutil.copy(join(filePath, file), join(MERGED_DATA_PATH, fold, category, file))