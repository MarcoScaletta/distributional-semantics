import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

# Preprocessa il file eseguendo il lemming con POS tagging salvando nel lemma risultato:
# Preprocessa i documenti
#   1. Eseguendo il POS TAGGING
#   2. Facendo il lemming sfruttando il POS TAGGING
#   3. Eliminando le stopwords
#   4. Salvando i lemmi insieme al POS TAG


# ex: "Luigi sono caduto" -> "(Luigi, n), (è,v), (caduto,v)" -> "cadere-v"
class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stopwords.add("%")
        self.stopwords.add("mr")
        self.stopwords.add("miss")
        self.stopwords.add("mrs")

    def init_doc_tag(self, doc):
        words_tags = nltk.pos_tag(word_tokenize(doc))
        initialized_doc = []
        for (word, tag) in words_tags:
            wn_postag = self.wn_pos(tag)
            if wn_postag != 'X' and not word in self.stopwords and word[
                    0].isalpha():
                initialized_doc += [
                    str(
                        self.lemmatizer.lemmatize(word, wn_postag) + "-" +
                        wn_postag)
                ]
        return initialized_doc

    @staticmethod
    def wn_pos(tag):
        if tag.startswith('NN'):
            return wn.NOUN
        elif tag.startswith('VB'):
            return wn.VERB
        elif tag.startswith('JJ'):
            return wn.ADJ
        elif tag.startswith('RB'):
            return wn.ADV
        else:
            return 'X'
