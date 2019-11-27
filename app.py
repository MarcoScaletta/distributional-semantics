import clusterizer
import preprocessor
from sklearn.metrics.pairwise import cosine_similarity
import keywordextractor
import multiplefilereader

directories = [
    "sport1", "sport2", "politics1", "business2", "business1", "politics2",
    "tech1", "tech2"
]

pairs_for_similarity = [(0, 1), (2, 5), (3, 4), (6, 7), (0, 2)]
docs = []
MultiReader = multiplefilereader.MultipleFileReader()

for directory in directories:
    docs += [MultiReader.readFilesFromDirectory(directory)]

Clusterizer = clusterizer.Clusterizer(preprocessor.Preprocessor())

clustering, model, feature_names = Clusterizer.cluster_texts(docs, 4)

similarity_matrix = cosine_similarity(model)
keywords_index = keywordextractor.KeywordExtractor().find_all_keywords(
    model, n_keywords=6)
keywords = dict()
for doc_id in keywords_index:
    keywords[directories[doc_id]] = [
        feature_names[index] for index in keywords_index[doc_id]
    ]

print()
print("Recupero le keywords per ogni documento")
print()
for doc in keywords:
    print("\t", doc, keywords[doc])
print()
print("Recupero le similarità per alcuni documenti")
print()
for i, j in pairs_for_similarity:
    sim = similarity_matrix[i][j]
    print("\tsim(" + directories[i] + "," + directories[j] + ") =", sim)
print()
print("Stampo i cluster")
print()
for c in clustering:
    print("\tCluster ", [directories[index] for index in clustering[c]])
