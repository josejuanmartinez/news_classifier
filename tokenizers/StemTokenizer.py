from nltk import WordNetLemmatizer, word_tokenize
from sklearn.feature_extraction import stop_words, re
from nltk.stem.porter import *


class StemTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, articles):
        result = list()
        for t in word_tokenize(articles):
            stem = self.stemmer.stem(t)
            # if lemma not in stop_words.ENGLISH_STOP_WORDS and re.search(r'^[a-zA-Z]+$', lemma) is not None:
            # if lemma not in stop_words.ENGLISH_STOP_WORDS:
            result.append(stem)
        return result