from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import collections


class Clusterizer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def cluster_texts(self, texts, clusters=2):

        vectorizer = CountVectorizer(tokenizer=self.preprocessor.init_doc_tag,
                                     max_df=0.8,
                                     min_df=0.1,
                                     lowercase=True)

        tfidf_model = vectorizer.fit_transform(texts)

        km_model = KMeans(n_clusters=clusters)
        km_model.fit(tfidf_model)

        clustering = collections.defaultdict(list)

        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)

        return clustering, tfidf_model, vectorizer.get_feature_names()
