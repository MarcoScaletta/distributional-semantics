
import clusterizer
import preprocessor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse
import keywordextractor
import multiplefilereader


directories = ["sport1", "sport2", "business1", "business2", "politics1", "politics2", "tech1", "tech2"]

docs = []
MultiReader = multiplefilereader.MultipleFileReader()

for directory in directories:
    docs += [MultiReader.readFilesFromDirectory(directory)]
    
Clusterizer = clusterizer.Clusterizer(preprocessor.Preprocessor())

clustering, model, feature_names= Clusterizer.cluster_texts(docs, 4)

similarity_matrix = cosine_similarity(model)
keywords_index = keywordextractor.KeywordExtractor().find_all_keywords(model)
keywords = dict()

for doc_id in keywords_index:
    keywords[directories[doc_id]] = [feature_names[index] for index in keywords_index[doc_id]]
for doc in keywords:
    print(doc, keywords[doc])
print()
for i in range(len(directories)-1):
    if i % 2 == 0:
        j = i+ 1
        sim = similarity_matrix[i][j]
        print("sim("+ directories[i] +","+directories[j]+") =", sim)
print()
for c in clustering:
    print("Cluster ", clustering[c])



