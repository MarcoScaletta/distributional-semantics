
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import collections
import preprocessor
from numpy import dot
from numpy.linalg import norm
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

from sklearn.metrics.pairwise import cosine_similarity



class Clusterizer:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


    def cluster_texts(self, texts, clusters=2):

        vectorizer = CountVectorizer(tokenizer= self.preprocessor.init_doc_tag,
                                    max_df=0.7,
                                    min_df=0.1,
                                    lowercase=True)

        tfidf_model = vectorizer.fit_transform(texts)
        doc_vecs = []

        for doc in list(tfidf_model.toarray()):
            doc_vecs += [list(doc)]

        km_model = KMeans(n_clusters=clusters)
        km_model.fit(tfidf_model)
    
        clustering = collections.defaultdict(list)
    
        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)
    
        return clustering, tfidf_model, vectorizer.get_feature_names()
