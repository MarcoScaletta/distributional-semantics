import numpy as np
from  scipy import sparse

class KeywordExtractor:

    def find_all_keywords(self, matrix, n_keywords=3):
        keyword_dict = dict()
        for i in range(matrix.shape[0]):
            keyword_dict[i] = self.find_keywords_index((matrix[i]).toarray()[0], n_keywords)
        return keyword_dict

    def find_keywords_index(self, vector, n_keywords=1):
        return list(vector.argsort()[-n_keywords:][::-1])
